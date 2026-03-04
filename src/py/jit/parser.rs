// src/py/jit/parser.rs
//! Expression AST and Pratt parser used by the JIT compiler.


/// Tokenizes a short expression string into individual symbols.
pub fn tokenize(expr: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut chars = expr.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_whitespace() {
            if !cur.is_empty() {
                tokens.push(cur.clone());
                cur.clear();
            }
            continue;
        }
        // handle two-character operators
        if c == '*' && chars.peek() == Some(&'*') {
            if !cur.is_empty() {
                tokens.push(cur.clone());
                cur.clear();
            }
            chars.next();
            tokens.push("**".to_string());
            continue;
        }
        if (c == '<' || c == '>' || c == '=' || c == '!') && chars.peek() == Some(&'=') {
            if !cur.is_empty() {
                tokens.push(cur.clone());
                cur.clear();
            }
            let mut op = c.to_string();
            op.push('=');
            chars.next();
            tokens.push(op);
            continue;
        }
        if "+-*/(),%<>=!".contains(c) {
            if !cur.is_empty() {
                tokens.push(cur.clone());
                cur.clear();
            }
            tokens.push(c.to_string());
            continue;
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

/// Simple expression AST for compiler.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Const(f64),
    Var(String),
    BinOp(Box<Expr>, String, Box<Expr>),
    Call(String, Vec<Expr>),
    UnaryOp(char, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    SumFor {
        iter_var: String,
        start: Box<Expr>,
        end: Box<Expr>,
        step: Option<Box<Expr>>,
        body: Box<Expr>,
        pred: Option<Box<Expr>>,
    },
    /// generator over a runtime container (e.g. Python list/ndarray)
    SumOver {
        iter_var: String,
        container: Box<Expr>,
        body: Box<Expr>,
        pred: Option<Box<Expr>>,
    },
}

/// Pratt parser implementation with exponent precedence and comparisons.
pub struct Parser {
    tokens: Vec<String>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<String>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn next(&mut self) -> Option<String> {
        if self.pos < self.tokens.len() {
            let s = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(s)
        } else {
            None
        }
    }

    pub fn parse_expr(&mut self) -> Option<Expr> {
        let mut node = self.parse_or()?;
        if let Some(tok) = self.peek() {
            if tok == "if" {
                self.next();
                let cond = self.parse_expr()?;
                if self.next()? != "else" {
                    return None;
                }
                let alt = self.parse_expr()?;
                node = Expr::Ternary(Box::new(cond), Box::new(node), Box::new(alt));
            }
        }
        Some(node)
    }

    fn parse_or(&mut self) -> Option<Expr> {
        let mut node = self.parse_and()?;
        while let Some(op) = self.peek() {
            if op == "or" {
                let op_str = self.next().unwrap();
                let rhs = self.parse_and()?;
                node = Expr::BinOp(Box::new(node), op_str, Box::new(rhs));
                continue;
            }
            break;
        }
        Some(node)
    }

    fn parse_and(&mut self) -> Option<Expr> {
        let mut node = self.parse_not()?;
        while let Some(op) = self.peek() {
            if op == "and" {
                let op_str = self.next().unwrap();
                let rhs = self.parse_not()?;
                node = Expr::BinOp(Box::new(node), op_str, Box::new(rhs));
                continue;
            }
            break;
        }
        Some(node)
    }

    fn parse_not(&mut self) -> Option<Expr> {
        if matches!(self.peek(), Some("not")) {
            self.next();
            let inner = self.parse_not()?;
            return Some(Expr::UnaryOp('!', Box::new(inner)));
        }
        self.parse_relation()
    }

    /// parse relational comparisons which have the lowest precedence
    /// (below arithmetic).  We call `parse_sum` so that `a + b < c` is
    /// interpreted as `(a + b) < c`.
    fn parse_relation(&mut self) -> Option<Expr> {
        let mut node = self.parse_sum()?;
        let mut chain: Vec<(String, Expr)> = Vec::new();
        while let Some(op) = self.peek() {
            if op == "<" || op == ">" || op == "<=" || op == ">=" || op == "==" || op == "!=" {
                let op_str = self.next().unwrap();
                let rhs = self.parse_sum()?;
                chain.push((op_str, rhs));
                continue;
            }
            break;
        }

        if chain.is_empty() {
            return Some(node);
        }

        // Python-style comparison chaining:
        // a < b < c  ==>  (a < b) and (b < c)
        let (first_op, first_rhs) = chain[0].clone();
        let mut result = Expr::BinOp(Box::new(node.clone()), first_op, Box::new(first_rhs.clone()));
        let mut prev_rhs = first_rhs;
        for (op, rhs) in chain.into_iter().skip(1) {
            let cmp = Expr::BinOp(Box::new(prev_rhs.clone()), op, Box::new(rhs.clone()));
            result = Expr::BinOp(Box::new(result), "and".to_string(), Box::new(cmp));
            prev_rhs = rhs;
        }
        node = result;
        Some(node)
    }

    /// parse addition and subtraction
    fn parse_sum(&mut self) -> Option<Expr> {
        let mut node = self.parse_term()?;
        while let Some(op) = self.peek() {
            if op == "+" || op == "-" {
                let op_str = self.next().unwrap();
                let rhs = self.parse_term()?;
                node = Expr::BinOp(Box::new(node), op_str, Box::new(rhs));
                continue;
            }
            break;
        }
        Some(node)
    }

    fn parse_power(&mut self) -> Option<Expr> {
        let mut node = self.parse_factor()?;
        while let Some(op) = self.peek() {
            if op == "**" {
                self.next();
                let rhs = self.parse_power()?; // right-associative
                node = Expr::BinOp(Box::new(node), "**".to_string(), Box::new(rhs));
                continue;
            }
            break;
        }
        Some(node)
    }

    fn parse_term(&mut self) -> Option<Expr> {
        let mut node = self.parse_power()?;
        while let Some(op) = self.peek() {
            if op == "*" || op == "/" || op == "%" {
                let op_str = self.next().unwrap();
                let rhs = self.parse_power()?;
                node = Expr::BinOp(Box::new(node), op_str, Box::new(rhs));
                continue;
            }
            break;
        }
        Some(node)
    }

    fn parse_factor(&mut self) -> Option<Expr> {
        if let Some(tok) = self.peek() {
            if tok == "(" {
                self.next();
                let expr = self.parse_expr();
                // only consume a closing parenthesis if it's actually there;
                // in generator expressions the `for` token may follow, so we
                // must not eat it.
                if matches!(self.peek(), Some(")")) {
                    self.next();
                }
                return expr;
            }
            if tok == "-" {
                // unary minus
                self.next();
                if let Some(e) = self.parse_factor() {
                    return Some(Expr::UnaryOp('-', Box::new(e)));
                }
            }
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Option<Expr> {
        if let Some(tok) = self.next() {
            if let Ok(num) = tok.parse::<f64>() {
                return Some(Expr::Const(num));
            }
            if tok == "True" {
                return Some(Expr::Const(1.0));
            }
            if tok == "False" {
                return Some(Expr::Const(0.0));
            }
            // identifier or function call
            if let Some(peek) = self.peek() {
                if peek == "(" {
                    // function call
                    self.next(); // consume '('
                    if tok == "sum" && !matches!(self.peek(), Some(")")) {
                        let body_expr = self.parse_expr()?;
                        if matches!(self.peek(), Some("for")) {
                            self.next(); // for
                            let iter_var = self.next()?;
                            let in_kw = self.next()?;
                            if in_kw != "in" {
                                return None;
                            }
                            // look ahead to see if this is a range() or a container
                            if matches!(self.peek(), Some("range")) {
                                let _range_kw = self.next()?;
                                let open = self.next()?;
                                if open != "(" {
                                    return None;
                                }
                                let first = self.parse_expr()?;
                                // parse up to three range arguments
                                let (start, end, step) = if matches!(self.peek(), Some(",")) {
                                    self.next(); // comma
                                    let second = self.parse_expr()?;
                                    if matches!(self.peek(), Some(",")) {
                                        self.next();
                                        let third = self.parse_expr()?;
                                        (first, second, Some(third))
                                    } else {
                                        (first, second, None)
                                    }
                                } else {
                                    (Expr::Const(0.0), first, None)
                                };
                                if !matches!(self.peek(), Some(")")) {
                                    return None;
                                }
                                self.next(); // inner ')'
                                // optional predicate after range
                                let mut pred: Option<Box<Expr>> = None;
                                if matches!(self.peek(), Some("if")) {
                                    self.next();
                                    pred = Some(Box::new(self.parse_expr()?));
                                }
                                if !matches!(self.peek(), Some(")")) {
                                    return None;
                                }
                                self.next(); // outer ')'
                                return Some(Expr::SumFor {
                                    iter_var,
                                    start: Box::new(start),
                                    end: Box::new(end),
                                    step: step.map(Box::new),
                                    body: Box::new(body_expr),
                                    pred,
                                });
                            } else {
                                // container form: parse single expr for container
                                let container = self.parse_or()?;
                                let mut pred: Option<Box<Expr>> = None;
                                if matches!(self.peek(), Some("if")) {
                                    self.next();
                                    pred = Some(Box::new(self.parse_expr()?));
                                }
                                if !matches!(self.peek(), Some(")")) {
                                    return None;
                                }
                                self.next(); // closing ')'
                                return Some(Expr::SumOver {
                                    iter_var,
                                    container: Box::new(container),
                                    body: Box::new(body_expr),
                                    pred,
                                });
                            }
                        } else {
                            let mut args = vec![body_expr];
                            while let Some(p) = self.peek() {
                                if p == ")" {
                                    self.next();
                                    break;
                                }
                                if p != "," {
                                    return None;
                                }
                                self.next();
                                if matches!(self.peek(), Some(")")) {
                                    self.next();
                                    break;
                                }
                                args.push(self.parse_expr()?);
                            }
                            return Some(Expr::Call(tok, args));
                        }
                    }

                    let mut args = Vec::new();
                    while let Some(p) = self.peek() {
                        if p == ")" {
                            self.next();
                            break;
                        }
                        args.push(self.parse_expr()?);
                        if matches!(self.peek(), Some(",")) {
                            self.next();
                        }
                    }
                    return Some(Expr::Call(tok, args));
                }
            }
            return Some(Expr::Var(tok));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_and_parse_simple() {
        let tokens = tokenize("a + b * 3");
        let mut p = Parser::new(tokens);
        let expr = p.parse_expr().unwrap();
        if let Expr::BinOp(_, op, _) = expr {
            assert_eq!(op, "+");
        } else {
            panic!("unexpected parse result");
        }
    }

    #[test]
    fn parse_sum_with_extra_parens() {
        let expr = "sum((i * i for i in range(n)))";
        let tokens = tokenize(expr);
        let mut p = Parser::new(tokens);
        let ast = p.parse_expr().expect("should parse");
        // expecting SumFor node
        match ast {
            Expr::Call(_, _) => panic!("generator parsed as regular call"),
            Expr::SumFor {..} => {}
            other => panic!("unexpected AST: {:?}", other),
        }
    }

    #[test]
    fn parse_boolean_precedence() {
        let tokens = tokenize("x < y and y < z or k");
        let mut p = Parser::new(tokens);
        let ast = p.parse_expr().expect("should parse");
        match ast {
            Expr::BinOp(_, op, _) => assert_eq!(op, "or"),
            other => panic!("unexpected AST: {:?}", other),
        }
    }

    #[test]
    fn parse_not_precedence() {
        let tokens = tokenize("not x < y and z");
        let mut p = Parser::new(tokens);
        let ast = p.parse_expr().expect("should parse");
        match ast {
            Expr::BinOp(lhs, op, _) => {
                assert_eq!(op, "and");
                match *lhs {
                    Expr::UnaryOp('!', _) => {}
                    other => panic!("expected not-unary on lhs, got: {:?}", other),
                }
            }
            other => panic!("unexpected AST: {:?}", other),
        }
    }

    #[test]
    fn parse_boolean_literals() {
        let mut p_true = Parser::new(tokenize("True"));
        assert_eq!(p_true.parse_expr(), Some(Expr::Const(1.0)));

        let mut p_false = Parser::new(tokenize("False"));
        assert_eq!(p_false.parse_expr(), Some(Expr::Const(0.0)));
    }

    #[test]
    fn parse_comparison_chain() {
        let tokens = tokenize("a < b < c");
        let mut p = Parser::new(tokens);
        let ast = p.parse_expr().expect("should parse");
        match ast {
            Expr::BinOp(_, op, _) => assert_eq!(op, "and"),
            other => panic!("unexpected AST: {:?}", other),
        }
    }
}
