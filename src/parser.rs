// parser

// src/parser.rs

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use pest::prec_climber::{Assoc, Operator, PrecClimber};

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct TypoParser;

lazy_static::lazy_static! {
    static ref PREC_CLIMBER: PrecClimber<Rule> = PrecClimber::new(vec![
        Operator::new(Rule::operator, Assoc::Left),
    ]);
}

struct Error{
    line: usize,
    message: String,
}

impl Error {
    fn new(line: usize, message: String) -> Error {
        Error{line, message}
    }
    fn handle(&self) { // returns an error log, similar to python
        println!("ERROR: ln {} :: {} ", self.line, self.message);
    }
    fn at(&self) {
        println!("ERROR at ln {} :: {} ", self.line, self.message);
    }
}

pub(crate) fn parse(source: &str) {
    match TypoParser::parse(Rule::program, source) {
        Ok(pairs) => {
            println!("Parsed successfully!\n");
            for pair in pairs {
                println!("{:#?}", pair);
                push_tokens(pair);
            }
        }
        Err(err) => {
            eprintln!("Parse Error: {}", err);
        }
    }
}

fn push_tokens(pair: Pair<Rule>) {
    println!("Rule: {:?}, Text: {}", pair.as_rule(), pair.as_str());
    for inner_pair in pair.into_inner() {
        push_tokens(inner_pair);
    }
}

fn eval_expr(pair: Pair<Rule>) -> String {
    PREC_CLIMBER.climb(
        pair.into_inner(),
        |primary| match primary.as_rule() {
            Rule::term => primary.as_str().to_string(),
            Rule::literal => primary.as_str().to_string(),
            Rule::IDENT => primary.as_str().to_string(),
            _ => format!("UNHANDLED({})", primary),
        },
        |lhs, op, rhs| format!("({} {} {})", lhs, op.as_str(), rhs),
    )
}
