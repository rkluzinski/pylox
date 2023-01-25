import argparse
import sys
from dataclasses import dataclass


class Token:
    def __init__(self, type, lexeme, literal, line):
        self.type = type
        self.lexeme = lexeme
        self.literal = literal
        self.line = line

    def __str__(self):
        return f"{self.type} {self.lexeme} {self.literal}"


class Scanner:
    KEYWORDS = {
        "and",
        "class",
        "else",
        "false",
        "for",
        "fun",
        "if",
        "nil",
        "or",
        "print",
        "return",
        "super",
        "this",
        "true",
        "var",
        "while",
    }

    def __init__(self, source):
        self.source = source
        self.start = 0
        self.current = 0
        self.line = 1
        self.tokens = []

    def scan_tokens(self):
        while self.current < len(self.source):
            self.start = self.current
            self.scan_token()
        self.tokens.append(Token("EOF", "", None, self.line))
        return self.tokens

    def scan_token(self):
        match c := self.advance():
            case "(": self.add_token("LEFT_PAREN")
            case ")": self.add_token("RIGHT_PAREN"),
            case "{": self.add_token("LEFT_BRACE"),
            case "}": self.add_token("RIGHT_BRACE"),
            case ",": self.add_token("COMMA"),
            case ".": self.add_token("DOT"),
            case "-": self.add_token("MINUS"),
            case "+": self.add_token("PLUS"),
            case ";": self.add_token("SEMICOLON"),
            case "*": self.add_token("STAR"),
            case "!": self.add_token("BANG_EQUAL" if self.match("=") else "BANG"),
            case "=": self.add_token("EQUAL_EQUAL" if self.match("=") else "EQUAL"),
            case "<": self.add_token("LESS_EQUAL" if self.match("=") else "LESS"),
            case ">": self.add_token("GREATER_EQUAL" if self.match("=") else "GREATER"),
            case "/": self.comment() if self.match("/") else self.add_token("SLASH")
            case " " | "\r" | "\t": pass
            case "\n": self.line += 1
            case "\"": self.string()
            case _:
                if c.isdigit():
                    self.number()
                elif c.isalpha():
                    self.identifier()
                else:
                    PyLox.error(self.line, "Unexpected character.")

    def comment(self):
        while self.peek() != "\n":
            self.current += 1

    def string(self):
        while self.peek() != "\"":
            if self.peek() == "\n":
                self.line += 1
            self.current += 1

        if self.at_end():
            PyLox.error(self.line, "Unterminated string.")
            return

        self.current += 1  # Closing "
        value = self.source[self.start + 1: self.current - 1]
        self.add_token("STRING", value)

    def number(self):
        while self.peek().isdigit():
            self.current += 1

        if self.peek() == ".":
            self.current += 1
            while self.peek().isdigit():
                self.current += 1

        value = float(self.source[self.start:self.current])
        self.add_token("NUMBER", value)

    def identifier(self):
        while self.peek().isalnum() or self.peek() == "_":
            self.current += 1

        text = self.source[self.start:self.current]
        if text in Scanner.KEYWORDS:
            self.add_token(text.upper())
        else:
            self.add_token("IDENTIFIER")

    def add_token(self, type, literal=None):
        lexeme = self.source[self.start:self.current]
        self.tokens.append(Token(type, lexeme, literal, self.line))

    def match(self, expected):
        if not self.at_end():
            if self.source[self.current] == expected:
                self.current += 1
                return True
        return False

    def advance(self):
        c = self.source[self.current]
        self.current += 1
        return c

    def peek(self):
        if self.at_end():
            return '\0'
        return self.source[self.current]

    def at_end(self):
        return not self.current < len(self.source)


def make_syntax_tree_node(base_class, name, *attrs):
    def __init__(self, *values):
        if len(values) != len(attrs):
            message = f"{name}.__init__() take {len(attrs)} positional arguments but {len(values)} were given"
            raise TypeError(message)

        for attr, value in zip(attrs, values):
            setattr(self, attr, value)

    visit_fn_name = f"visit_{name.lower()}_{base_class.__name__.lower()}"

    def accept(self, visitor):
        return getattr(visitor, visit_fn_name)(self)

    subclass = type(
        name, (base_class,),
        {"__init__": __init__, "accept": accept})

    setattr(base_class, name, subclass)

    def visit(self, expr):
        raise NotImplementedError()

    setattr(base_class.Visitor, visit_fn_name, visit)


class Expr:
    def accept(self, visitor):
        raise NotImplementedError()

    class Visitor:
        pass


class Stmt:
    def accept(self, visitor):
        raise NotImplementedError()

    class Visitor:
        pass


# Expr subclasses
make_syntax_tree_node(Expr, "Assign", "name", "value")
make_syntax_tree_node(Expr, "Binary", "left", "operator", "right")
make_syntax_tree_node(Expr, "Grouping", "expression")
make_syntax_tree_node(Expr, "Literal", "value")
make_syntax_tree_node(Expr, "Logical", "left", "operator", "right")
make_syntax_tree_node(Expr, "Unary", "operator", "right")
make_syntax_tree_node(Expr, "Variable", "name")

# Stmt subclasses
make_syntax_tree_node(Stmt, "Block", "statements")
make_syntax_tree_node(Stmt, "Expression", "expression")
make_syntax_tree_node(Stmt, "If", "condition", "then_branch", "else_branch")
make_syntax_tree_node(Stmt, "Print", "expression")
make_syntax_tree_node(Stmt, "Var", "name", "initializer")
make_syntax_tree_node(Stmt, "While", "condition", "body")


class Parser:
    class Error(RuntimeError):
        pass

    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0

    def parse(self):
        statements = []
        while not self.at_end():
            statements.append(self.declaration())
        return statements

    def declaration(self):
        try:
            if self.match("VAR"):
                return self.var_declaration()
            return self.statement()
        except Parser.Error as error:
            self.synchronize()
            return None

    def statement(self):
        if self.match("FOR"):
            return self.for_statement()
        if self.match("IF"):
            return self.if_statement()
        if self.match("PRINT"):
            return self.print_statement()
        if self.match("LEFT_BRACE"):
            return Stmt.Block(self.block())
        if self.match("WHILE"):
            return self.while_statement()
        return self.expression_statement()

    def block(self):
        statements = []
        while self.peek().type != "RIGHT_BRACE" and not self.at_end():
            statements.append(self.declaration())
        self.consume("RIGHT_BRACE", "Expected '}' after block.")
        return statements

    def for_statement(self):
        self.consume("LEFT_PAREN", "Expected '(' after for.")

        initializer = None
        if self.match("VAR"):
            initializer = self.var_declaration()
        elif not self.match("SEMICOLON"):
            initializer = self.expression_statement()

        condition = None
        if not self.match("SEMICOLON"):
            condition = self.expression()
        self.consume("SEMICOLON", "Expected ';' after loop condition.")

        increment = None
        if not self.match("SEMICOLON"):
            increment = self.expression()

        self.consume("RIGHT_PAREN", "Expected ')' after for clauses.")
        body = self.statement()

        if increment:
            body = Stmt.Block([body, increment])

        if not condition:
            condition = Expr.Literal(True)

        body = Stmt.While(condition, body)

        if initializer:
            body = Stmt.Block([initializer, body])

        return body

    def if_statement(self):
        self.consume("LEFT_PAREN", "Expected '(' after if.")
        condition = self.expression()
        self.consume("RIGHT_PAREN", "Expected ')' after condition.")
        then_branch = self.statement()
        else_branch = None
        if self.match("ELSE"):
            else_branch = self.statement()
        return Stmt.If(condition, then_branch, else_branch)

    def expression_statement(self):
        expression = self.expression()
        self.consume("SEMICOLON", "Expected ';' after expression.")
        return Stmt.Expression(expression)

    def print_statement(self):
        expression = self.expression()
        self.consume("SEMICOLON", "Expected ';' after expression.")
        return Stmt.Print(expression)

    def var_declaration(self):
        name = self.consume("IDENTIFIER", "Expected identifier.")
        initializer = None
        if self.match("EQUAL"):
            initializer = self.expression()
        self.consume("SEMICOLON", "Expected ';' after variable declaration.")
        return Stmt.Var(name, initializer)

    def while_statement(self):
        self.consume("LEFT_PAREN", "Expected '(' after while.")
        condition = self.expression()
        self.consume("RIGHT_PAREN", "Expected ')' after condition.")
        body = self.statement()
        return Stmt.While(condition, body)

    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.logic_or()
        if equals := self.match("EQUAL"):
            value = self.assignment()
            if isinstance(expr, Expr.Variable):
                return Expr.Assign(expr.name, value)
            self.error(equals, "Invalid assignment target.")
        return expr

    def logic_or(self):
        expr = self.logic_and()
        while operator := self.match("OR"):
            expr = Expr.Logical(expr, operator, self.logic_and())
        return expr

    def logic_and(self):
        expr = self.equality()
        while operator := self.match("AND"):
            expr = Expr.Logical(expr, operator, self.equality())
        return expr

    def equality(self):
        expr = self.comparison()
        while operator := self.match("BANG_EQUAL", "EQUAL_EQUAL"):
            expr = Expr.Binary(expr, operator, self.comparison())
        return expr

    def comparison(self):
        expr = self.term()
        while operator := self.match("GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL"):
            expr = Expr.Binary(expr, operator, self.term())
        return expr

    def term(self):
        expr = self.factor()
        while operator := self.match("MINUS", "PLUS"):
            expr = Expr.Binary(expr, operator, self.factor())
        return expr

    def factor(self):
        expr = self.unary()
        while operator := self.match("SLASH", "STAR"):
            expr = Expr.Binary(expr, operator, self.unary())
        return expr

    def unary(self):
        if operator := self.match("BANG", "MINUS"):
            return Expr.Unary(operator, self.unary())
        return self.primary()

    def primary(self):
        if self.match("FALSE"):
            return Expr.Literal(False)
        if self.match("TRUE"):
            return Expr.Literal(True)
        if self.match("NIL"):
            return Expr.Literal(None)
        if token := self.match("NUMBER", "STRING"):
            return Expr.Literal(token.literal)
        if self.match("LEFT_PAREN"):
            expr = self.expression()
            self.consume("RIGHT_PAREN", "Expected ')' after expression.")
            return Expr.Grouping(expr)
        if token := self.match("IDENTIFIER"):
            return Expr.Variable(token)
        raise self.error(self.peek(), "Expected expression.")

    def synchronize(self):
        while not self.at_end():
            match self.peek().type:
                case "SEMICOLON":
                    self.advance()
                    return
                case "CLASS" | "FUN" | "VAR" | "FOR" | "IF" | "WHILE" | "PRINT" | "RETURN":
                    return
            self.advance()

    def consume(self, token_type, message):
        if token := self.match(token_type):
            return token
        raise self.error(self.peek(), message)

    def match(self, *token_types):
        if self.peek().type in token_types:
            return self.advance()
        return None

    def advance(self):
        token = self.peek()
        if not self.at_end():
            self.current += 1
        return token

    def at_end(self):
        return self.peek().type == "EOF"

    def peek(self):
        return self.tokens[self.current]

    def error(self, token, message):
        if token.type == "EOF":
            PyLox.report(token.line, " at end", message)
        else:
            PyLox.report(token.line, f" at '{token.lexeme}'", message)
        return Parser.Error()


class Environment:
    def __init__(self, enclosing=None):
        self.values = {}
        self.enclosing = enclosing

    def define(self, name, value):
        self.values[name.lexeme] = value

    def assign(self, name, value):
        if name.lexeme in self.values:
            self.values[name.lexeme] = value
            return
        if self.enclosing:
            self.enclosing.assign(name, value)
            return
        raise Interpreter.Error(
            name, f"Undefined variable '{name.lexeme}'.")

    def get(self, name):
        if name.lexeme in self.values:
            return self.values[name.lexeme]
        if self.enclosing:
            return self.enclosing.get(name)
        raise Interpreter.Error(
            name, f"Undefined variable '{name.lexeme}'.")


class Interpreter(Expr.Visitor, Stmt.Visitor):
    class Error(RuntimeError):
        def __init__(self, token, message):
            self.token = token
            self.message = message

    def __init__(self):
        self.environment = Environment()

    def interpret(self, stmts):
        try:
            for stmt in stmts:
                self.execute(stmt)
        except Interpreter.Error as error:
            PyLox.runtime_error(error)

    def stringify(self, object):
        if object is None:
            return "nil"
        text = str(object)
        if isinstance(object, float) and text[-2:] == ".0":
            text = text[:-2]
        return text

    def evaluate(self, expr):
        return expr.accept(self)

    def execute(self, stmt):
        return stmt.accept(self)

    def execute_block(self, statements, environment):
        try:
            self.environment = environment
            for statement in statements:
                self.execute(statement)
        finally:
            self.environment = environment.enclosing

    def visit_block_stmt(self, stmt):
        self.execute_block(stmt.statements, Environment(self.environment))

    def visit_expression_stmt(self, stmt):
        self.evaluate(stmt.expression)

    def visit_if_stmt(self, stmt):
        if self.is_truthy(self.evaluate(stmt.condition)):
            self.execute(stmt.then_branch)
        elif stmt.else_branch:
            self.execute(stmt.else_branch)

    def visit_print_stmt(self, stmt):
        value = self.evaluate(stmt.expression)
        print(self.stringify(value))

    def visit_var_stmt(self, stmt):
        value = None
        if stmt.initializer:
            value = self.evaluate(stmt.initializer)
        self.environment.define(stmt.name, value)

    def visit_while_stmt(self, stmt):
        while self.is_truthy(self.evaluate(stmt.condition)):
            self.execute(stmt.body)

    def visit_assign_expr(self, expr):
        value = self.evaluate(expr.value)
        self.environment.assign(expr.name, value)
        return value

    def visit_binary_expr(self, expr):
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)
        operator = expr.operator
        match operator.type:
            case "BANG_EQUAL": return not self.is_equal(left, right)
            case "EQUAL_EQUAL": return self.is_equal(left, right)
            case "GREATER":
                self.check_operands(operator, float, left, right)
                return left > right
            case "GREATER_EQUAL":
                self.check_operands(operator, float, left, right)
                return left >= right
            case "LESS":
                self.check_operands(operator, float, left, right)
                return left < right
            case "LESS_EQUAL":
                self.check_operands(operator, float, left, right)
                return left <= right
            case "MINUS":
                self.check_operands(operator, float, left, right)
                return left - right
            case "PLUS":
                if isinstance(left, float) and isinstance(right, float):
                    return left + right
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
                raise Interpreter.Error(
                    operator, "Operands must be numbers or strings.")
            case "SLASH":
                self.check_operands(operator, float, left, right)
                if right == 0.0:
                    raise Interpreter.Error(
                        operator, "Cannot divide by zero.")
                return left / right
            case "STAR":
                self.check_operands(operator, float, left, right)
                return left * right
            case _:
                return None

    def visit_grouping_expr(self, expr):
        return self.evaluate(expr.expression)

    def visit_literal_expr(self, expr):
        return expr.value

    def visit_logical_expr(self, expr):
        left = self.evaluate(expr.left)
        if expr.operator.type == "OR":
            if self.is_truthy(left):
                return left
        else:
            if not self.is_truthy(left):
                return left
        return self.evaluate(expr.right)

    def visit_unary_expr(self, expr):
        right = self.evaluate(expr.right)
        match expr.operator.type:
            case "BANG": return not self.is_truthy(right)
            case "MINUS":
                self.check_operands(expr.operator, float, right)
                return -right
            case _: return None

    def visit_variable_expr(self, expr):
        return self.environment.get(expr.name)

    def is_truthy(self, object):
        if object is None:
            return False
        if isinstance(object, bool):
            return object
        return True

    def is_equal(self, left, right):
        return left == right

    def check_operands(self, operator, expected, *operands):
        if any(not isinstance(operand, expected) for operand in operands):
            if len(operands) == 1:
                raise Interpreter.Error(
                    operator, "Operand must be a number.")
            raise Interpreter.Error(
                operator, "Operands must be numbers.")


class PyLox:
    interpreter = Interpreter()
    had_error = False
    had_runtime_error = False

    @staticmethod
    def main(args):
        if args.filename is not None:
            PyLox.run_file(args.filename)
        else:
            PyLox.run_prompt()

        if PyLox.had_error:
            exit(65)
        if PyLox.had_runtime_error:
            exit(70)

    @staticmethod
    def run_file(filename):
        with open(filename, "r") as file:
            PyLox.run(file.read())

    @staticmethod
    def run_prompt():
        while True:
            try:
                line = input("> ")
            except EOFError:
                print()
                break
            PyLox.had_error = False
            PyLox.had_runtime_error = False
            PyLox.run(line)

    @staticmethod
    def run(source):
        scanner = Scanner(source)
        tokens = scanner.scan_tokens()

        parser = Parser(tokens)
        statements = parser.parse()

        if PyLox.had_error:
            return

        PyLox.interpreter.interpret(statements)

    @staticmethod
    def error(line, message):
        PyLox.report(line, "", message)

    @staticmethod
    def runtime_error(error):
        print(f"{error.message} [line {error.token.line}]", file=sys.stderr)
        PyLox.had_runtime_error = True

    @staticmethod
    def report(line, where, message):
        print(f"[line {line}] Error{where}: {message}", file=sys.stderr)
        PyLox.had_error = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Pylox", description="Run Lox scripts")
    parser.add_argument("filename", nargs="?")
    args = parser.parse_args()
    PyLox().main(args)
