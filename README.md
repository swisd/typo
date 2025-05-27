# typo
extreme pain language that requires you to define anything and everything


<!-- sample ig lol -->

```
program helloworld;

typ INT i32;
typ STR str;
typ FLOAT f64;

var pi: FLOAT &mut = 3.14159;

// you can also use the builtin types
var myVariable: str = "hello";

args add {
    define a as INT;
    define b as INT;
}
args sub {
    define a as INT;
    define b as INT;
}
args mul {
    define a as INT;
    define b as INT;
}
args div {
    define a as INT;
    define b as INT;
}

repr Math {
    return: INT;
}

cls Math {
    fn add(a, b) -> INT {
        return a + b;
    }
    fn sub(a, b) -> INT {
        return a + b;
    }
    fn mul(a, b) -> INT {
        return a + b;
    }
    fn div(a, b) -> INT {
        return a + b;
    }
}


fn main() -> None {
    prn("Hello World");
    prn(Math::add(5, 10));
}
```
