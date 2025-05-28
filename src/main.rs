use pest::Parser; // This imports the Parser *trait* for use in `eval_expr` etc.
use pest_derive::Parser; // This imports the Parser *derive macro*
use pest::iterators::{Pair, Pairs};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
extern crate regex;
use regex::Regex;
use lazy_static::lazy_static;
use std::io::{self, Write};
use std::fs;

// --- Pest Grammar Setup ---
// The `#[derive(Parser)]` macro generates the parser code from src/grammar.pest
#[derive(Parser)]
#[grammar = "grammar.pest"] // Path to your grammar file
struct MathParser;

// --- AST Definitions (Same as before) ---

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Assignment(Ident, Expr),
    FunctionDef(Ident, Ident, Expr), // Func name, param name, body
    Expr(Expr), // A bare expression statement
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Number(f64),
    Var(Ident),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    UnaryMinus(Box<Expr>),
    FuncCall(Ident, Vec<Expr>),
    Paren(Box<Expr>), // This will likely be optimized out during AST building
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

pub type Ident = String;

// --- Custom Error Type (Same as before) ---

#[derive(Debug)]
pub enum EvalError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    TypeError(String),
    ArityMismatch(String, usize, usize), // func_name, expected, found
    MathError(String),
    InvalidArguments(String),
    ParseError(String), // New error for parsing
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EvalError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
            EvalError::UndefinedFunction(name) => write!(f, "Unknown function: {}", name),
            EvalError::TypeError(msg) => write!(f, "Type error: {}", msg),
            EvalError::ArityMismatch(name, expected, found) => write!(f, "Function '{}' expected {} arguments, but got {}", name, expected, found),
            EvalError::MathError(msg) => write!(f, "Math error: {}", msg),
            EvalError::InvalidArguments(msg) => write!(f, "Invalid arguments: {}", msg),
            EvalError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl Error for EvalError {}

// --- Differentiated Math Functions (dmath.rs) ---
// (Same as before, keep it here or in a separate module)
pub mod dmath {
    pub fn dsin(x: f64) -> f64 { x.sin() }
    pub fn dcos(x: f64) -> f64 { x.cos() }
    pub fn dtan(x: f64) -> f64 { x.tan() }
    pub fn dsinh(x: f64) -> f64 { x.sinh() }
    pub fn dcosh(x: f64) -> f64 { x.cosh() }
    pub fn dtanh(x: f64) -> f64 { x.tanh() }
    pub fn dasin(x: f64) -> f64 { x.asin() }
    pub fn dacos(x: f64) -> f64 { x.acos() }
    pub fn datan(x: f64) -> f64 { x.atan() }
}

// --- Differential Functions (differential.rs) ---
// (Same as before, keep it here or in a separate module)
pub mod differential {
    pub fn dadd(a: f64, b: f64) -> f64 { a + b }
    pub fn dsub(a: f64, b: f64) -> f64 { a - b }
    pub fn dmul(a: f64, b: f64) -> f64 { a * b }
    pub fn ddiv(a: f64, b: f64) -> f64 { a / b }
    pub fn dpow(a: f64, b: f64) -> f64 { a.powf(b) }
}


// --- Global State and Built-ins (Same as before) ---
lazy_static! {
    static ref VARS_TABLE: std::sync::Mutex<HashMap<Ident, f64>> = std::sync::Mutex::new(HashMap::new());
    static ref FUNCS_TABLE: std::sync::Mutex<HashMap<Ident, (Ident, Expr)>> = std::sync::Mutex::new(HashMap::new());

    static ref CONSTANTS: HashMap<Ident, f64> = {
        let mut m = HashMap::new();
        m.insert("pi".to_string(), std::f64::consts::PI);
        m.insert("e".to_string(), std::f64::consts::E);
        m.insert("inf".to_string(), std::f64::INFINITY);
        m.insert("nan".to_string(), std::f64::NAN);
        m.insert("tau".to_string(), std::f64::consts::TAU);
        m
    };

    static ref BUILTINS: HashMap<Ident, Box<dyn Fn(&[f64]) -> Result<f64, EvalError> + Send + Sync>> = {
        let mut m: HashMap<Ident, Box<dyn Fn(&[f64]) -> Result<f64, EvalError> + Send + Sync>> = HashMap::new();

        m.insert("sin".to_string(), Box::new(|args| match args { [x] => Ok(dmath::dsin(*x)), _ => Err(EvalError::ArityMismatch("sin".to_string(), 1, args.len())), }));
        m.insert("cos".to_string(), Box::new(|args| match args { [x] => Ok(dmath::dcos(*x)), _ => Err(EvalError::ArityMismatch("cos".to_string(), 1, args.len())), }));
        m.insert("tan".to_string(), Box::new(|args| match args { [x] => Ok(dmath::dtan(*x)), _ => Err(EvalError::ArityMismatch("tan".to_string(), 1, args.len())), }));
        m.insert("log".to_string(), Box::new(|args| match args { [x] => Ok(x.ln()), [x, base] => Ok(x.log(*base)), _ => Err(EvalError::InvalidArguments("log expects 1 or 2 arguments".to_string())), }));
        m.insert("trunc".to_string(), Box::new(|args| match args { [x] => Ok(x.trunc()), _ => Err(EvalError::ArityMismatch("trunc".to_string(), 1, args.len())), }));
        m.insert("round".to_string(), Box::new(|args| match args { [x] => Ok(x.round()), _ => Err(EvalError::ArityMismatch("round".to_string(), 1, args.len())), }));
        m.insert("cbrt".to_string(), Box::new(|args| match args { [x] => Ok(x.cbrt()), _ => Err(EvalError::ArityMismatch("cbrt".to_string(), 1, args.len())), }));
        m.insert("sqrt".to_string(), Box::new(|args| match args { [x] => Ok(x.sqrt()), _ => Err(EvalError::ArityMismatch("sqrt".to_string(), 1, args.len())), }));
        m.insert("ceil".to_string(), Box::new(|args| match args { [x] => Ok(x.ceil()), _ => Err(EvalError::ArityMismatch("ceil".to_string(), 1, args.len())), }));
        m.insert("floor".to_string(), Box::new(|args| match args { [x] => Ok(x.floor()), _ => Err(EvalError::ArityMismatch("floor".to_string(), 1, args.len())), }));
        m.insert("fact".to_string(), Box::new(|args| match args { [x] if *x >= 0.0 && x.fract() == 0.0 => { let n = *x as u64; Ok((1..=n).product::<u64>() as f64) }, _ => Err(EvalError::InvalidArguments("fact expects a non-negative integer".to_string())), }));
        m.insert("log2".to_string(), Box::new(|args| match args { [x] => Ok(x.log2()), _ => Err(EvalError::ArityMismatch("log2".to_string(), 1, args.len())), }));
        m.insert("log10".to_string(), Box::new(|args| match args { [x] => Ok(x.log10()), _ => Err(EvalError::ArityMismatch("log10".to_string(), 1, args.len())), }));
        m.insert("sinh".to_string(), Box::new(|args| match args { [x] => Ok(dmath::dsinh(*x)), _ => Err(EvalError::ArityMismatch("sinh".to_string(), 1, args.len())), }));
        m.insert("cosh".to_string(), Box::new(|args| match args { [x] => Ok(dmath::dcosh(*x)), _ => Err(EvalError::ArityMismatch("cosh".to_string(), 1, args.len())), }));
        m.insert("tanh".to_string(), Box::new(|args| match args { [x] => Ok(dmath::dtanh(*x)), _ => Err(EvalError::ArityMismatch("tanh".to_string(), 1, args.len())), }));
        m.insert("sin1".to_string(), Box::new(|args| match args { [x] => Ok(dmath::dasin(*x)), _ => Err(EvalError::ArityMismatch("sin1".to_string(), 1, args.len())), }));
        m.insert("cos1".to_string(), Box::new(|args| match args { [x] => Ok(dmath::dacos(*x)), _ => Err(EvalError::ArityMismatch("cos1".to_string(), 1, args.len())), }));
        m.insert("tan1".to_string(), Box::new(|args| match args { [x] => Ok(dmath::datan(*x)), _ => Err(EvalError::ArityMismatch("tan1".to_string(), 1, args.len())), }));
        m.insert("max".to_string(), Box::new(|args| match args { [a, b] => Ok(a.max(*b)), _ => Err(EvalError::ArityMismatch("max".to_string(), 2, args.len())), }));
        m.insert("min".to_string(), Box::new(|args| match args { [a, b] => Ok(a.min(*b)), _ => Err(EvalError::ArityMismatch("min".to_string(), 2, args.len())), }));
        m.insert("dadd".to_string(), Box::new(|args| match args { [a, b] => Ok(differential::dadd(*a, *b)), _ => Err(EvalError::ArityMismatch("dadd".to_string(), 2, args.len())), }));
        m.insert("dsub".to_string(), Box::new(|args| match args { [a, b] => Ok(differential::dsub(*a, *b)), _ => Err(EvalError::ArityMismatch("dsub".to_string(), 2, args.len())), }));
        m.insert("dmul".to_string(), Box::new(|args| match args { [a, b] => Ok(differential::dmul(*a, *b)), _ => Err(EvalError::ArityMismatch("dmul".to_string(), 2, args.len())), }));
        m.insert("ddiv".to_string(), Box::new(|args| match args { [a, b] => Ok(differential::ddiv(*a, *b)), _ => Err(EvalError::ArityMismatch("ddiv".to_string(), 2, args.len())), }));
        m.insert("dexp".to_string(), Box::new(|args| match args { [a, b] => Ok(differential::dpow(*a, *b)), _ => Err(EvalError::ArityMismatch("dexp".to_string(), 2, args.len())), }));
        m
    };
}

// --- Expression Evaluator (Same as before) ---
fn eval_expr(expr: &Expr, local_vars: &HashMap<Ident, f64>) -> Result<f64, EvalError> {
    // none
}


// --- AST Building from Pest Pairs ---
// This is where you convert Pest's generic parse tree into your strongly-typed AST.
fn parse_expr(pair: Pair<Rule>) -> Result<Expr, EvalError> {
    // none
}
// --- REPL and File Runner (Mostly same, but with Pest parsing) ---

fn preprocess_line(mut line: String) -> String {
    line
}

fn repl(line: &str) {
    if line.trim().is_empty() {
        return;
    }
    if line.trim() == "exit" || line.trim() == "quit" {
        std::process::exit(0);
    }

    let preprocessed_line = preprocess_line(line.to_string());
}

fn run_file(filename: &str) -> Result<(), Box<dyn Error>> {
    let contents = fs::read_to_string(filename)?;
    for line in contents.lines() {
        repl(line);
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        // File mode
        match run_file(&args[1]) {
            Ok(_) => {}
            Err(e) => eprintln!("File processing error: {}", e),
        }
        let _ = io::stdin().read_line(&mut String::new()); // Keep console open
        std::process::exit(0);
    } else {
        // REPL mode
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        loop {
            print!("> ");
            stdout.flush().unwrap();

            let mut line = String::new();
            match stdin.read_line(&mut line) {
                Ok(0) => break, // EOF (Ctrl+D)
                Ok(_) => repl(&line),
                Err(e) => {
                    eprintln!("Input error: {}", e);
                    break;
                }
            }
        }
    }
}