# ===----------------------------------------------------------------------=== #
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implement mojo dynamic dispatch using trait and union type.

Mojo generic can only support same type `T` as parameter. How to achieve subtype dynamic
dispatch, such as `std::box::Boxed<dyn trait>` in rust, or `std::shared_ptr<Base *>` in c++?

We can combine the idea of Variant and trait in Mojo to achieve this.

A Variant is a sum type, such as `enum` in rust, or `tagged union` in c++ (or std::variant
 in c++17). It can hold a value that could take on several different, but fixed types.

We can define several struct implementing one same trait, such as `Echoable`. And define a
Variant from all these types, which is constrained by one same trait. Then we implement
the trait `Echoable` for the Variant.

For example:

```mojo

struct MyVariant[*Ts: Echoable](
    Echoable,
):
    ...
    fn echo(self) -> String:
        var res: String = String("")
        var cur = self._get_state()[]

        @parameter
        fn each[i: Int]():
            if cur == i:
                alias T = Ts[i]
                res = self[T].echo()

        unroll[each, len(VariadicList(Ts))]()

        return res
```

As the above code shows, when we have a Variant at runtime, we known its underlying type.
We can compare it with all the candidate types, and then get the actual object, call the
trait function. This has exactly the same effect of dynamic dispatch in rust or c++.

The comparing between types is auctually comparing using an `int8`, which is determined by
the order of types in compile time. And loop of the types is unrolled in compile time using
`@parameter` meta programming. At runtime, there are just some `int8` comparing to determing
which object to use. Only Exactly one of them will be true. I'm not sure how fast it is
comparing to vtable in c++ or rust, but it should be really fast with only some `int8` comparing.

Wow, awesome mojo! We just implement dynamic dispatch in a library, not in the compiler! Just
as mojo's design principle said, implement the language in library, not in compiler.

Right now we cannot use trait as generic parameters. So Variant cannot be reused directly.
Most of code are same as implementation in mojo `stdlib/src/utils/variant.mojo`, so I just
copied the code. Maybe when macro are supported in the future, we can have a more generic
implementation.


You can use the same idea to implement your dynamic dispatch (like subtype in c++).

For example:

```mojo

alias AOrB = MyVariant[A, B]
var a = A()
var b = B()

var x = AOrB(a)
var y = AOrB(b)

# result: this is A
print(x.echo())

# result: this is B
print(y.echo())
```
"""

from python import Python

from sys import alignof, sizeof
from sys.intrinsics import _type_is_eq

from memory import UnsafePointer
from memory.unsafe_pointer import (
    initialize_pointee_move,
    move_from_pointee,
    move_pointee,
)

from utils import unroll

@always_inline
fn align_up(value: Int, alignment: Int) -> Int:
    var div_ceil = (value + alignment - 1)._positive_div(alignment)
    return div_ceil * alignment


# ===----------------------------------------------------------------------=== #
# Variant
# ===----------------------------------------------------------------------=== #

trait Echoable(CollectionElement, ExplicitlyCopyable):
    fn echo(self) -> String:
        ...


# FIXME(#27380): Can't pass *Ts to a function parameter, only type parameter.
struct UnionSize[*Ts: Echoable]():
    @staticmethod
    fn compute() -> Int:
        var size = 0

        @parameter
        fn each[i: Int]():
            size = max(size, align_up(sizeof[Ts[i]](), alignof[Ts[i]]()))

        unroll[each, len(VariadicList(Ts))]()
        return align_up(size, alignof[Int]())


# FIXME(#27380): Can't pass *Ts to a function parameter, only type parameter.
struct UnionTypeIndex[T: Echoable, *Ts: Echoable]:
    @staticmethod
    fn compute() -> Int8:
        var result = -1

        @parameter
        fn each[i: Int]():
            alias q = Ts[i]

            @parameter
            if _type_is_eq[q, T]():
                result = i

        unroll[each, len(VariadicList(Ts))]()
        return result


struct MyVariant[*Ts: Echoable](
    Echoable,
):
    """A runtime-variant type.

    Data for this type is stored internally. Currently, its size is the
    largest size of any of its variants plus a 16-bit discriminant.

    You can
        - use `isa[T]()` to check what type a variant is
        - use `unsafe_take[T]()` to take a value from the variant
        - use `[T]` to get a value out of a variant
            - This currently does an extra copy/move until we have lifetimes
            - It also temporarily requires the value to be mutable
        - use `set[T](owned new_value: T)` to reset the variant to a new value

    Example:
    ```mojo
    from utils import Variant
    alias IntOrString = Variant[Int, String]
    fn to_string(inout x: IntOrString) -> String:
        if x.isa[String]():
            return x[String][]
        # x.isa[Int]()
        return str(x[Int][])

    # They have to be mutable for now, and implement Echoable
    var an_int = IntOrString(4)
    var a_string = IntOrString(String("I'm a string!"))
    var who_knows = IntOrString(0)
    import random
    if random.random_ui64(0, 1):
        who_knows.set[String]("I'm actually a string too!")

    print(to_string(an_int))
    print(to_string(a_string))
    print(to_string(who_knows))
    ```

    Parameters:
      Ts: The elements of the variadic.
    """

    # Fields
    alias _sentinel: Int = -1
    alias _mlir_type = __mlir_type[
        `!kgen.variant<[rebind(:`, __type_of(Ts), ` `, Ts, `)]>`
    ]
    var _impl: Self._mlir_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(inout self, *, unsafe_uninitialized: ()):
        """Unsafely create an uninitialized Variant.

        Args:
            unsafe_uninitialized: Marker argument indicating this initializer is unsafe.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, Self._mlir_type]

    fn __init__[T: Echoable](inout self, owned value: T):
        """Create a variant with one of the types.

        Parameters:
            T: The type to initialize the variant to. Generally this should
                be able to be inferred from the call type, eg. `Variant[Int, String](4)`.

        Args:
            value: The value to initialize the variant with.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._mlir_type]
        self._get_state()[] = Self._check[T]()
        initialize_pointee_move(self._get_ptr[T](), value^)

    fn __init__(inout self, other: Self):
        """Explicitly creates a deep copy of an existing variant.

        Args:
            other: The value to copy from.
        """
        self = Self(unsafe_uninitialized=())
        self._get_state()[] = other._get_state()[]

        @parameter
        fn each[i: Int]():
            if self._get_state()[] == i:
                alias T = Ts[i]
                initialize_pointee_move(
                    UnsafePointer.address_of(self._impl).bitcast[T](),
                    UnsafePointer.address_of(other._impl).bitcast[T]()[],
                )

        unroll[each, len(VariadicList(Ts))]()

    fn __copyinit__(inout self, other: Self):
        """Creates a deep copy of an existing variant.

        Args:
            other: The variant to copy from.
        """

        # Delegate to explicit copy initializer.
        self = Self(other=other)

    fn __moveinit__(inout self, owned other: Self):
        """Move initializer for the variant.

        Args:
            other: The variant to move.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._mlir_type]
        self._get_state()[] = other._get_state()[]

        @parameter
        fn each[i: Int]():
            if self._get_state()[] == i:
                alias T = Ts[i]
                # Calls the correct __moveinit__
                move_pointee(src=other._get_ptr[T](), dst=self._get_ptr[T]())

        unroll[each, len(VariadicList(Ts))]()

    fn __del__(owned self):
        """Destroy the variant."""
        self._call_correct_deleter()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __getitem__[
        T: Echoable
    ](self: Reference[Self, _, _]) -> ref [self.lifetime] T:
        """Get the value out of the variant as a type-checked type.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            The internal data represented as a `Reference[T]`.
        """
        if not self[].isa[T]():
            abort("get: wrong variant type")

        return self[].unsafe_get[T]()[]

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn _get_ptr[T: Echoable](self) -> UnsafePointer[T]:
        constrained[
            Self._check[T]() != Self._sentinel, "not a union element type"
        ]()
        return UnsafePointer.address_of(self._impl).bitcast[T]()

    fn _get_state(
        self: Reference[Self, _, _]
    ) -> Reference[Int8, self.is_mutable, self.lifetime]:
        var int8_self = UnsafePointer(self).bitcast[Int8]()
        return (int8_self + UnionSize[Ts].compute())[]

    @always_inline
    fn _call_correct_deleter(inout self):
        @parameter
        fn each[i: Int]():
            if self._get_state()[] == i:
                alias q = Ts[i]
                destroy_pointee(self._get_ptr[q]().address)

        unroll[each, len(VariadicList(Ts))]()

    @always_inline
    fn take[T: Echoable](inout self) -> T:
        """Take the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        Parameters:
            T: The type to take out.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        if not self.isa[T]():
            abort("taking the wrong type!")

        return self.unsafe_take[T]()

    @always_inline
    fn unsafe_take[T: Echoable](inout self) -> T:
        """Unsafely take the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            T: The type to take out.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        debug_assert(self.isa[T](), "taking wrong type")
        # don't call the variant's deleter later
        self._get_state()[] = Self._sentinel
        return move_from_pointee(self._get_ptr[T]())

    @always_inline
    fn replace[
        Tin: Echoable, Tout: Echoable
    ](inout self, value: Tin) -> Tout:
        """Replace the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        Parameters:
            Tin: The type to put in.
            Tout: The type to take out.

        Args:
            value: The value to put in.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        if not self.isa[Tout]():
            abort("taking out the wrong type!")

        return self.unsafe_replace[Tin, Tout](value)

    @always_inline
    fn unsafe_replace[
        Tin: Echoable, Tout: Echoable
    ](inout self, value: Tin) -> Tout:
        """Unsafely replace the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            Tin: The type to put in.
            Tout: The type to take out.

        Args:
            value: The value to put in.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        debug_assert(self.isa[Tout](), "taking out the wrong type!")

        var x = self.unsafe_take[Tout]()
        self.set[Tin](value)
        return x^

    fn set[T: Echoable](inout self, owned value: T):
        """Set the variant value.

        This will call the destructor on the old value, and update the variant's
        internal type and data to the new value.

        Parameters:
            T: The new variant type. Must be one of the Variant's type arguments.

        Args:
            value: The new value to set the variant to.
        """
        self = Self(value^)

    fn isa[T: Echoable](self) -> Bool:
        """Check if the variant contains the required type.

        Parameters:
            T: The type to check.

        Returns:
            True if the variant contains the requested type.
        """
        alias idx = Self._check[T]()
        return self._get_state()[] == idx

    fn unsafe_get[
        T: Echoable
    ](self: Reference[Self, _, _]) -> Reference[
        T, self.is_mutable, self.lifetime
    ]:
        """Get the value out of the variant as a type-checked type.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            The internal data represented as a `Reference[T]`.
        """
        debug_assert(self[].isa[T](), "get: wrong variant type")
        return self[]._get_ptr[T]()[]

    @staticmethod
    fn _check[T: Echoable]() -> Int8:
        return UnionTypeIndex[T, Ts].compute()

    fn echo(self) -> String:
        var res: String = String("")
        var cur = self._get_state()[]

        @parameter
        fn each[i: Int]():
            if cur == i:
                alias T = Ts[i]
                res = self[T].echo()

        unroll[each, len(VariadicList(Ts))]()

        return res


@value
struct A(
    Echoable,
):
    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#
    fn __init__(inout self, other: Self):
        pass

    fn echo(self) -> String:
        return "this is A"

@value
struct B(
    Echoable
):
    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#
    fn __init__(inout self, other: Self):
        pass

    fn echo(self) -> String:
        return "this is B"

@value
struct C(
    Echoable
):
    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#
    fn __init__(inout self, other: Self):
        pass

    fn echo(self) -> String:
        return "this is C"

fn test_dyn():
    alias AOrBOrC = MyVariant[A, B, C]
    var a = A()
    var b = B()
    var c = C()

    var x = AOrBOrC(a)
    var y = AOrBOrC(b)
    var z = AOrBOrC(c)

    print(x.echo())
    print(y.echo())
    print(z.echo())

fn main() raises:
    test_dyn()
