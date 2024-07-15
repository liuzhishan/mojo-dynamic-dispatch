# mojo-dynamic-dispatch

Implement mojo dynamic dispatch using trait and union type.

## Implementation

Mojo generic can only support same type `T` as parameter. How to achieve subtype dynamic
dispatch, such as `std::box::Boxed<dyn trait>` in rust, or `std::shared_ptr<Base *>` ?

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


## Usaage

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
