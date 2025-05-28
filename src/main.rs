// TYPO compiler


mod interp;
mod parser;

fn main() -> Result<(), String> {
    let raw_file: String = std::fs::read_to_string("test.typo").expect("Cannot read TYPO file");
    let ast_root: Vec<Box<AstNode>> = typo::parser::parse(source: &raw_file).expect("unsuccessful parse");
    interp::interp(ast_root)?;
    Ok(())
}