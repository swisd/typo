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
    match expr {
        Expr::Number(n) => Ok(*n),
        Expr::Var(name) => {
            if let Some(val) = local_vars.get(name) {
                Ok(*val)
            } else if let Some(val) = VARS_TABLE.lock().unwrap().get(name) {
                Ok(*val)
            } else if let Some(val) = CONSTANTS.get(name) {
                Ok(*val)
            } else {
                Err(EvalError::UndefinedVariable(name.clone()))
            }
        }
        Expr::BinOp(left, op, right) => {
            let l_val = eval_expr(left, local_vars)?;
            let r_val = eval_expr(right, local_vars)?;
            match op {
                BinOp::Add => Ok(l_val + r_val),
                BinOp::Sub => Ok(l_val - r_val),
                BinOp::Mul => Ok(l_val * r_val),
                BinOp::Div => {
                    if r_val == 0.0 {
                        Err(EvalError::MathError("Division by zero".to_string()))
                    } else {
                        Ok(l_val / r_val)
                    }
                }
                BinOp::Pow => Ok(l_val.powf(r_val)),
            }
        }
        Expr::UnaryMinus(expr) => Ok(-eval_expr(expr, local_vars)?),
        Expr::FuncCall(name, args_exprs) => {
            let args: Result<Vec<f64>, EvalError> = args_exprs.iter()
                .map(|arg_expr| eval_expr(arg_expr, local_vars))
                .collect();
            let args = args?; // Propagate error if any arg evaluation fails

            if let Some(func) = BUILTINS.get(name) {
                func(&args)
            } else if let Some((param_name, body)) = FUNCS_TABLE.lock().unwrap().get(name) {
                if args.len() != 1 {
                    return Err(EvalError::ArityMismatch(name.clone(), 1, args.len()));
                }
                let mut new_locals = local_vars.clone(); // Clone to extend with param
                new_locals.insert(param_name.clone(), args[0]);
                eval_expr(body, &new_locals)
            } else {
                Err(EvalError::UndefinedFunction(name.clone()))
            }
        }
        Expr::Paren(expr) => eval_expr(expr, local_vars),
    }
}


// --- AST Building from Pest Pairs ---
// This is where you convert Pest's generic parse tree into your strongly-typed AST.
fn parse_expr(pair: Pair<Rule>) -> Result<Expr, EvalError> {
    let expr = match pair.as_rule() {
        // These are the top-level expression rules, they operate on the current 'pair' directly
        Rule::expr | Rule::sum => {
            let mut terms = pair.into_inner();
            let mut expr = parse_expr(terms.next().unwrap())?;

            while let Some(op) = terms.next() {
                let rhs = parse_expr(terms.next().unwrap())?;
                expr = match op.as_rule() {
                    Rule::PLUS => Expr::BinOp(Box::new(expr), BinOp::Add, Box::new(rhs)),
                    Rule::MINUS => Expr::BinOp(Box::new(expr), BinOp::Sub, Box::new(rhs)),
                    _ => unreachable!(),
                };
            }
            expr
        },
        Rule::product => {
            let mut factors = pair.into_inner();
            let mut expr = parse_expr(factors.next().unwrap())?;

            while let Some(op) = factors.next() {
                let rhs = parse_expr(factors.next().unwrap())?;
                expr = match op.as_rule() {
                    Rule::TIMES => Expr::BinOp(Box::new(expr), BinOp::Mul, Box::new(rhs)),
                    Rule::DIV => Expr::BinOp(Box::new(expr), BinOp::Div, Box::new(rhs)),
                    _ => unreachable!(),
                };
            }
            expr
        },
        Rule::power => {
            let mut base_and_exp = pair.into_inner();
            let base = parse_expr(base_and_exp.next().unwrap())?;

            if let Some(_op) = base_and_exp.next() { // Changed to _op as it's not used
                let exponent = parse_expr(base_and_exp.next().unwrap())?;
                Expr::BinOp(Box::new(base), BinOp::Pow, Box::new(exponent))
            } else {
                base
            }
        },
        Rule::unary => {
            let mut inner = pair.into_inner();
            if inner.peek().map(|p| p.as_rule() == Rule::PLUS || p.as_rule() == Rule::MINUS).unwrap_or(false) {
                let op = inner.next().unwrap();
                let expr = parse_expr(inner.next().unwrap())?;
                match op.as_rule() {
                    Rule::MINUS => Expr::UnaryMinus(Box::new(expr)),
                    Rule::PLUS => expr,
                    _ => unreachable!(),
                }
            } else {
                parse_expr(inner.next().unwrap())?
            }
        },
        // **THIS IS THE CRITICAL CHANGE AREA**
        // Rule::primary directly contains its children, so we get the inner pair FIRST.
        Rule::primary => {
            // Get the single inner rule that 'primary' contains
            let primary_inner_pair = pair.into_inner().next().unwrap(); // consumes 'pair' here

            match primary_inner_pair.as_rule() { // Now match on this inner pair
                Rule::NUMBER => Expr::Number(primary_inner_pair.as_str().parse::<f64>().map_err(|e| EvalError::ParseError(format!("Invalid number: {}", e)))?),
                Rule::IDENT => Expr::Var(primary_inner_pair.as_str().to_string()),
                Rule::function_call_expr => {
                    // Pass primary_inner_pair (which is Rule::function_call_expr) directly
                    // to a helper function, or handle it here if it's cleaner.
                    // Let's integrate the logic directly here for now, using primary_inner_pair
                    let mut call_parts = primary_inner_pair.into_inner(); // This consumes primary_inner_pair
                    let func_name = call_parts.next().unwrap().as_str().to_string(); // This gets IDENT `f`

                    let mut args = Vec::new();
                    for arg_pair in call_parts { // Iterate over the remaining children of function_call_expr
                        match arg_pair.as_rule() {
                            Rule::expr_list => {
                                for expr_in_list in arg_pair.into_inner() {
                                    args.push(parse_expr(expr_in_list)?);
                                }
                            },
                            Rule::expr => { // If it's a single expr
                                args.push(parse_expr(arg_pair)?);
                            },
                            _ => {
                                return Err(EvalError::ParseError(format!("Unexpected token in function arguments: {:?}", arg_pair.as_rule())));
                            }
                        }
                    }
                    Expr::FuncCall(func_name, args)
                },
                Rule::expr => Expr::Paren(Box::new(parse_expr(primary_inner_pair)?)), // Pass the inner expr
                _ => unreachable!(),
            }
        },
        // Direct matches for NUMBER and IDENT at the top level of `expr` (if they are direct children of an expression rule)
        // These are redundant if 'primary' always handles them, but good to keep if grammar allows direct matches.
        Rule::NUMBER => Expr::Number(pair.as_str().parse::<f64>().map_err(|e| EvalError::ParseError(format!("Invalid number: {}", e)))?),
        Rule::IDENT => Expr::Var(pair.as_str().to_string()),
        _ => {
            return Err(EvalError::ParseError(format!("Unexpected expression rule: {:?}", pair.as_rule())));
        }
    };
    Ok(expr)
}
// --- REPL and File Runner (Mostly same, but with Pest parsing) ---

lazy_static! {
    static ref IMPLICIT_MULT_REGEX1: Regex = Regex::new(r"(\d)([A-Za-z\(])").unwrap();
    static ref IMPLICIT_MULT_REGEX2: Regex = Regex::new(r"(\))(\d|[A-Za-z\(])").unwrap();
}

fn preprocess_line(mut line: String) -> String {
    line = IMPLICIT_MULT_REGEX1.replace_all(&line, r"$1*$2").to_string();
    line = IMPLICIT_MULT_REGEX2.replace_all(&line, r"$1*$2").to_string();
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

    // Pest parsing step
    match MathParser::parse(Rule::start, &preprocessed_line) {
        Ok(mut pairs) => {
            let start_pair = pairs.next().unwrap(); // This is `Rule::start`

            // Get the inner content of the 'start' rule, which should be the first statement.
            // For a REPL line, we generally expect only one statement.
            let outer_stmt_pair = start_pair.into_inner().next().unwrap(); // This is `Rule::statement`

            // Now, we need to match on the `outer_stmt_pair` which IS `Rule::statement`.
            // Then, its *inner* pair will be the actual assignment, func_def, or expr.
            match outer_stmt_pair.as_rule() {
                Rule::statement => { // Handle the Rule::statement itself
                    let actual_stmt_pair = outer_stmt_pair.into_inner().next().unwrap(); // This gets the inner specific statement type

                    match actual_stmt_pair.as_rule() { // Now match against the *actual* statement rule
                        Rule::assignment => {
                            let mut inner_pairs = actual_stmt_pair.into_inner();
                            let name = inner_pairs.next().unwrap().as_str().to_string(); // This gets IDENT "x"
                            // inner_pairs.next().unwrap(); // Consume the EQ sign -- Add this line!
                            let _eq_token = inner_pairs.next().unwrap(); // Consume the EQ token, no need to store it
                            let expr_pair = inner_pairs.next().unwrap(); // This now correctly gets the 'expr' Pair for "10"

                            match parse_expr(expr_pair) {
                                Ok(expr) => {
                                    match eval_expr(&expr, &HashMap::new()) {
                                        Ok(val) => {
                                            VARS_TABLE.lock().unwrap().insert(name.clone(), val);
                                            println!("= {}", val);
                                        },
                                        Err(e) => eprintln!("Error during assignment evaluation: {}", e),
                                    }
                                },
                                Err(e) => eprintln!("Error building AST: {}", e),
                            }
                        },
                        Rule::function_def => {
                            let mut inner_pairs = actual_stmt_pair.into_inner();
                            let func_name = inner_pairs.next().unwrap().as_str().to_string();
                            let param_name = inner_pairs.next().unwrap().as_str().to_string();
                            let body_expr_pair = inner_pairs.next().unwrap();
                            match parse_expr(body_expr_pair) {
                                Ok(body_expr) => {
                                    FUNCS_TABLE.lock().unwrap().insert(func_name.clone(), (param_name, body_expr));
                                    // No immediate evaluation for function definition
                                }
                                Err(e) => eprintln!("Error building AST for function body: {}", e),
                            }
                        }
                        Rule::expr => { // Bare expression statement
                            match parse_expr(actual_stmt_pair) { // Pass the expr pair directly
                                Ok(expr) => {
                                    match eval_expr(&expr, &HashMap::new()) {
                                        Ok(val) => println!("= {}", val),
                                        Err(e) => eprintln!("Error: {}", e),
                                    }
                                }
                                Err(e) => eprintln!("Error building AST: {}", e),
                            }
                        }
                        _ => eprintln!("Parsing Error: Unexpected specific statement type: {:?}", actual_stmt_pair.as_rule()),
                    }
                }
                _ => eprintln!("Parsing Error: Expected 'statement' rule (from start), found {:?}", outer_stmt_pair.as_rule()),
            }
        }
        Err(e) => eprintln!("Parsing Error: {}", e),
    }
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