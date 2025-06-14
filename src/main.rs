// TYPO compiler

use std::env;
use std::fs;
use crate::parser::parse;

mod interp;
mod parser;

fn main() {
    let input = fs::read_to_string("C:/RS/typo/src/pmi.typo").unwrap();
    parser::parse(&*input);
}