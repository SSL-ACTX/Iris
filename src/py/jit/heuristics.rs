// src/py/jit/heuristics.rs
//! Placeholder heuristics module for expression simplification and optimization.
//!
//! Currently this is a no-op; future improvements may perform constant folding,
//! algebraic simplifications, loop unrolling hints, branch prediction hints, etc.

use crate::py::jit::parser::Expr;

/// Apply lightweight transformations to an expression before code generation.
///
/// Right now it simply returns the input unchanged, but having a dedicated
/// module makes it easy to incrementally introduce more intelligence without
/// cluttering parser/codegen logic.
pub fn optimize(expr: Expr) -> Expr {
    // constant folding and minor simplifications
    match expr {
        Expr::BinOp(lhs, op, rhs) => {
            let lhs = optimize(*lhs);
            let rhs = optimize(*rhs);

            // constant folding first
            if let (Expr::Const(a), Expr::Const(b)) = (&lhs, &rhs) {
                let v = match op.as_str() {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => a / b,
                    "**" => a.powf(*b),
                    _ => return Expr::BinOp(Box::new(lhs), op, Box::new(rhs)),
                };
                return Expr::Const(v);
            }

            // algebraic simplifications
            match (op.as_str(), &lhs, &rhs) {
                ("+", Expr::Const(0.0), _) => return rhs.clone(),
                ("+", _, Expr::Const(0.0)) => return lhs.clone(),
                ("-", _, Expr::Const(0.0)) => return lhs.clone(),
                ("*", Expr::Const(0.0), _) => return Expr::Const(0.0),
                ("*", _, Expr::Const(0.0)) => return Expr::Const(0.0),
                ("*", Expr::Const(1.0), _) => return rhs.clone(),
                ("*", _, Expr::Const(1.0)) => return lhs.clone(),
                ("/", _, Expr::Const(1.0)) => return lhs.clone(),
                _ => {}
            }

            Expr::BinOp(Box::new(lhs), op, Box::new(rhs))
        }
        Expr::UnaryOp(c, expr) => {
            let expr = optimize(*expr);
            if let Expr::Const(a) = expr {
                if c == '-' {
                    Expr::Const(-a)
                } else {
                    Expr::UnaryOp(c, Box::new(Expr::Const(a)))
                }
            } else {
                Expr::UnaryOp(c, Box::new(expr))
            }
        }
        Expr::Ternary(cond, thenb, elseb) => {
            let cond = optimize(*cond);
            let thenb = optimize(*thenb);
            let elseb = optimize(*elseb);
            if let Expr::Const(c) = &cond {
                return if *c != 0.0 { thenb } else { elseb };
            }
            Expr::Ternary(Box::new(cond), Box::new(thenb), Box::new(elseb))
        }
        Expr::Call(name, args) => {
            Expr::Call(name, args.into_iter().map(optimize).collect())
        }
        Expr::SumFor { iter_var, start, end, body } => {
            let start = optimize(*start);
            let end = optimize(*end);
            let body = optimize(*body);

            // if bounds are constant we can sometimes reduce the loop
            if let (Expr::Const(a), Expr::Const(b)) = (&start, &end) {
                let len = b - a; // number of iterations (exclusive end)
                if len.is_finite() {
                    // case 1: body is just the loop variable -> arithmetic series
                    if body == Expr::Var(iter_var.clone()) {
                        // sum over i=a..b-1 = n*(a + (b-1))/2
                        let sum = len * (a + (b - 1.0)) / 2.0;
                        return Expr::Const(sum);
                    }
                    // case 2: body is constant -> constant*len
                    if let Expr::Const(c) = body {
                        return Expr::Const(c * len);
                    }
                }
            }

            Expr::SumFor {
                iter_var,
                start: Box::new(start),
                end: Box::new(end),
                body: Box::new(body),
            }
        },
        other => other,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::py::jit::parser::{tokenize, Parser};

    fn parse(expr: &str) -> Expr {
        Parser::new(tokenize(expr)).parse_expr().unwrap()
    }

    #[test]
    fn constant_folding_binops() {
        assert_eq!(optimize(parse("2 + 2")), Expr::Const(4.0));
        assert_eq!(optimize(parse("2 * 3")), Expr::Const(6.0));
        assert_eq!(optimize(parse("2 ** 3")), Expr::Const(8.0));
    }

    #[test]
    fn constant_folding_unary() {
        assert_eq!(optimize(parse("-5")), Expr::Const(-5.0));
    }

    #[test]
    fn no_fold_with_variable() {
        assert_eq!(optimize(parse("2 + x + 3")), parse("2 + x + 3"));
    }

    #[test]
    fn algebraic_simplifications() {
        assert_eq!(optimize(parse("x + 0")), parse("x"));
        assert_eq!(optimize(parse("0 + x")), parse("x"));
        assert_eq!(optimize(parse("x - 0")), parse("x"));
        assert_eq!(optimize(parse("x * 1")), parse("x"));
        assert_eq!(optimize(parse("1 * y")), parse("y"));
        assert_eq!(optimize(parse("y * 0")), Expr::Const(0.0));
        assert_eq!(optimize(parse("0 * y")), Expr::Const(0.0));
        assert_eq!(optimize(parse("x / 1")), parse("x"));
    }

    #[test]
    fn ternary_constants() {
        assert_eq!(optimize(parse("x if 1 else y")), parse("x"));
        assert_eq!(optimize(parse("x if 0 else y")), parse("y"));
    }

    #[test]
    fn sum_for_reduction() {
        assert_eq!(optimize(parse("sum(i for i in range(5))")), Expr::Const(10.0));
        assert_eq!(optimize(parse("sum(2 for i in range(3))")), Expr::Const(6.0));
        assert_eq!(optimize(parse("sum(i for i in range(2,5))")), Expr::Const(9.0));
    }
}
