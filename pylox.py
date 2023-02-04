import argparse
import sys
import time


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
make_syntax_tree_node(Expr, "Call", "callee", "paren", "arguments")
make_syntax_tree_node(Expr, "Get", "object", "name")
make_syntax_tree_node(Expr, "Grouping", "expression")
make_syntax_tree_node(Expr, "Literal", "value")
make_syntax_tree_node(Expr, "Logical", "left", "operator", "right")
make_syntax_tree_node(Expr, "Set", "object", "name", "value")
make_syntax_tree_node(Expr, "Super", "keyword", "method")
make_syntax_tree_node(Expr, "This", "keyword")
make_syntax_tree_node(Expr, "Unary", "operator", "right")
make_syntax_tree_node(Expr, "Variable", "name")

# Stmt subclasses
make_syntax_tree_node(Stmt, "Block", "statements")
make_syntax_tree_node(Stmt, "Class", "name", "superclass", "methods")
make_syntax_tree_node(Stmt, "Expression", "expression")
make_syntax_tree_node(Stmt, "Function", "name", "params", "body")
make_syntax_tree_node(Stmt, "If", "condition", "then_branch", "else_branch")
make_syntax_tree_node(Stmt, "Print", "expression")
make_syntax_tree_node(Stmt, "Return", "keyword", "value")
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
            if self.match("CLASS"):
                return self.class_declaration()
            if self.match("FUN"):
                return self.function("function")
            if self.match("VAR"):
                return self.var_declaration()
            return self.statement()
        except Parser.Error as error:
            self.synchronize()
            return None

    def class_declaration(self):
        name = self.consume("IDENTIFIER", "Expected class name.")

        superclass = None
        if self.match("LESS"):
            superclass = Expr.Variable(self.consume(
                "IDENTIFIER", "Expected superclass name."))

        self.consume("LEFT_BRACE", "Expected '{' before class bpdy.")

        methods = []
        while not self.at_end() and self.peek().type != "RIGHT_BRACE":
            methods.append(self.function("method"))

        self.consume("RIGHT_BRACE", "Expected '}' after class body.")
        return Stmt.Class(name, superclass, methods)

    def function(self, kind):
        name = self.consume("IDENTIFIER", f"Expected {kind} name.")
        self.consume("LEFT_PAREN", "Expected '(' after function name.")

        params = []
        if self.peek().type != "RIGHT_PAREN":
            params.append(self.consume(
                "IDENTIFIER", "Expected parameter name."))
            while self.match("COMMA"):
                if len(params) >= 255:
                    self.error(
                        self.peek(), "Can't have more than 255 parameters.")
                params.append(self.consume(
                    "IDENTIFIER", "Expected parameter name."))

        self.consume("RIGHT_PAREN", "Expected ')' after parameters.")
        self.consume("LEFT_BRACE", f"Expect '{'{'}' before {kind} body.")
        return Stmt.Function(name, params, self.block())

    def statement(self):
        if self.match("FOR"):
            return self.for_statement()
        if self.match("IF"):
            return self.if_statement()
        if self.match("PRINT"):
            return self.print_statement()
        if self.match("LEFT_BRACE"):
            return Stmt.Block(self.block())
        if keyword := self.match("RETURN"):
            return self.return_statement(keyword)
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

    def return_statement(self, keyword):
        value = None
        if not self.peek().type == "SEMICOLON":
            value = self.expression()
        self.consume("SEMICOLON", "Expected ';' after return value.")
        return Stmt.Return(keyword, value)

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
            elif isinstance(expr, Expr.Get):
                return Expr.Set(expr.object, expr.name, value)
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
        return self.call()

    def call(self):
        expr = self.primary()
        while True:
            if self.match("LEFT_PAREN"):
                expr = self.finish_call(expr)
            elif self.match("DOT"):
                name = self.consume(
                    "IDENTIFIER", "Expected property name after '.'.")
                expr = Expr.Get(expr, name)
            else:
                break
        return expr

    def finish_call(self, callee):
        arguments = []
        if self.peek().type != "RIGHT_PAREN":
            arguments.append(self.expression())
            while self.match("COMMA"):
                if len(arguments) >= 255:
                    self.error(
                        self.peek(), "Can't have more than 255 arguments.")
                arguments.append(self.expression())
        paren = self.consume("RIGHT_PAREN", "Expected ')' after arguments.")
        return Expr.Call(callee, paren, arguments)

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
        if keyword := self.match("SUPER"):
            self.consume("DOT", "Expected '.' after 'super'.")
            method = self.consume(
                "IDENTIFIER", "Expected superclass method name.")
            return Expr.Super(keyword, method)
        if keyword := self.match("THIS"):
            return Expr.This(keyword)
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


class Resolver(Expr.Visitor, Stmt.Visitor):
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.scopes = [{}]
        self.current_function = "NONE"
        self.current_class = "NONE"

    def resolve(self, expr_or_stmt):
        expr_or_stmt.accept(self)

    def visit_block_stmt(self, stmt):
        self.begin_scope()
        for statement in stmt.statements:
            self.resolve(statement)
        self.end_scope()

    def visit_class_stmt(self, stmt):
        enclosing_class = self.current_class
        self.current_class = "CLASS"

        self.declare(stmt.name)
        if stmt.superclass:
            self.current_class = "SUBCLASS"
            if stmt.name.lexeme == stmt.superclass.name.lexeme:
                PyLox.error(stmt.superclass.name.line,
                            "A class can't inherit from itself.")
            self.resolve(stmt.superclass)

        if stmt.superclass:
            self.begin_scope()
            self.scopes[-1]["super"] = True

        self.begin_scope()
        self.scopes[-1]["this"] = True
        for method in stmt.methods:
            kind = "METHOD"
            if method.name.lexeme == "init":
                kind = "INITIALIZER"
            self.resolve_function(method, kind)
        self.define(stmt.name)
        self.end_scope()

        if stmt.superclass:
            self.end_scope()

        self.current_class = enclosing_class

    def visit_expression_stmt(self, stmt):
        self.resolve(stmt.expression)

    def visit_function_stmt(self, stmt):
        self.declare(stmt.name)
        self.define(stmt.name)
        self.resolve_function(stmt, "FUNCTION")

    def visit_if_stmt(self, stmt):
        self.resolve(stmt.condition)
        self.resolve(stmt.then_branch)
        if stmt.else_branch:
            self.resolve(stmt.else_branch)

    def visit_print_stmt(self, stmt):
        self.resolve(stmt.expression)

    def visit_return_stmt(self, stmt):
        if self.current_function == "NONE":
            PyLox.error(stmt.keyword.line, "Can't return from top-level code.")
        if stmt.value:
            if self.current_function == "INITIALIZER":
                PyLox.error(stmt.keyword.line,
                            "Can't return a value from an initializer.")
            self.resolve(stmt.value)

    def visit_var_stmt(self, stmt):
        self.declare(stmt.name)
        if (stmt.initializer):
            self.resolve(stmt.initializer)
        self.define(stmt.name)

    def visit_while_stmt(self, stmt):
        self.resolve(stmt.condition)
        self.resolve(stmt.body)

    def visit_assign_expr(self, expr):
        self.resolve(expr.value)
        self.resolve_local(expr, expr.name)

    def visit_binary_expr(self, expr):
        self.resolve(expr.left)
        self.resolve(expr.right)

    def visit_call_expr(self, expr):
        self.resolve(expr.callee)
        for argument in expr.arguments:
            self.resolve(argument)

    def visit_get_expr(self, expr):
        self.resolve(expr.object)

    def visit_grouping_expr(self, expr):
        self.resolve(expr.expression)

    def visit_literal_expr(self, expr):
        pass

    def visit_logical_expr(self, expr):
        self.resolve(expr.left)
        self.resolve(expr.right)

    def visit_set_expr(self, expr):
        self.resolve(expr.value)
        self.resolve(expr.object)

    def visit_super_expr(self, expr):
        if self.current_class == "NONE":
            PyLox.error(expr.keyword.line,
                        "Can't use 'super' outside of a class.")
        elif self.current_class == "SUBCLASS":
            PyLox.error(expr.keyword.line,
                        "Can't use 'super' in a class with no superclass.")
        self.resolve_local(expr, expr.keyword)

    def visit_this_expr(self, expr):
        if self.current_class == "NONE":
            PyLox.error(expr.keyword.line,
                        "Can't use 'this' outside of a class.")
            return None
        self.resolve_local(expr, expr.keyword)

    def visit_unary_expr(self, expr):
        self.resolve(expr.right)

    def visit_variable_expr(self, expr):
        if self.scopes and self.scopes[-1].get(expr.name.lexeme, None) == False:
            PyLox.Error(
                expr.name.line, "Can't read local variable in its own initializer.")
        self.resolve_local(expr, expr.name)

    def begin_scope(self):
        self.scopes.append({})

    def end_scope(self):
        self.scopes.pop()

    def declare(self, name):
        if self.scopes:
            if name.lexeme in self.scopes[-1]:
                PyLox.error(
                    name.line, "Already a variable with this name in this scope.")
            self.scopes[-1][name.lexeme] = False

    def define(self, name):
        if self.scopes:
            self.scopes[-1][name.lexeme] = True

    def resolve_function(self, function, kind):
        enclosing = self.current_function
        self.current_function = kind
        self.begin_scope()
        for param in function.params:
            self.declare(param)
            self.define(param)
        for stmt in function.body:
            self.resolve(stmt)
        self.end_scope()
        self.current_function = enclosing

    def resolve_local(self, expr, name):
        for i, scope in enumerate(reversed(self.scopes)):
            if name.lexeme in scope:
                self.interpreter.resolve(expr, i)
                break


class Environment:
    def __init__(self, enclosing=None):
        self.values = {}
        self.enclosing = enclosing

    def define(self, name, value):
        self.values[name] = value

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

    def assign_at(self, distance, name, value):
        self.ancestor(distance).values[name] = value
        return value

    def get_at(self, distance, name):
        return self.ancestor(distance).values[name]

    def ancestor(self, distance):
        environment = self
        for i in range(distance):
            environment = environment.enclosing
        return environment


class LoxCallable:
    def arity(self):
        raise NotImplementedError()

    def call(self, interpreter, arguments):
        raise NotImplementedError()


class LoxFunction(LoxCallable):
    def __init__(self, declaration, closure, is_initializer):
        self.declaration = declaration
        self.closure = closure
        self.is_initializer = is_initializer

    def arity(self):
        return len(self.declaration.params)

    def bind(self, instance):
        environment = Environment(self.closure)
        environment.define("this", instance)
        return LoxFunction(self.declaration, environment, self.is_initializer)

    def call(self, interpreter, arguments):
        environment = Environment(self.closure)
        for param, argument in zip(self.declaration.params, arguments):
            environment.define(param.lexeme, argument)
        try:
            interpreter.execute_block(self.declaration.body, environment)
        except Interpreter.Return as returnValue:
            if self.is_initializer:
                return self.closure.get_at(0, "this")
            return returnValue.value

        if self.is_initializer:
            return self.closure.get_at(0, "this")

    def __str__(self):
        return f"<fn {self.declaration.name.lexeme}>"


class LoxClass(LoxCallable):
    def __init__(self, name, superclass, methods):
        self.name = name
        self.superclass = superclass
        self.methods = methods

    def arity(self):
        if initializer := self.find_method("init"):
            return initializer.arity()
        return 0

    def call(self, interpreter, arguments):
        instance = LoxInstance(self)
        if initializer := self.find_method("init"):
            initializer.bind(instance).call(interpreter, arguments)
        return instance

    def find_method(self, name):
        if method := self.methods.get(name, None):
            return method
        if self.superclass:
            return self.superclass.find_method(name)
        return None

    def __str__(self):
        return self.name


class LoxInstance:
    def __init__(self, klass):
        self.klass = klass
        self.fields = {}

    def get(self, name):
        if name.lexeme in self.fields:
            return self.fields[name.lexeme]
        if method := self.klass.find_method(name.lexeme):
            return method.bind(self)
        raise Interpreter.Error(
            name, f"Undefined property '{name.lexeme}'.")

    def set(self, name, value):
        self.fields[name.lexeme] = value

    def __str__(self):
        return f"{self.klass.name} instance"


class Interpreter(Expr.Visitor, Stmt.Visitor):
    class Error(RuntimeError):
        def __init__(self, token, message):
            self.token = token
            self.message = message

    class Return(RuntimeError):
        def __init__(self, value):
            self.value = value

    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals
        self.locals = {}

        def make_native_function(name, arity, call):
            return type(name, (LoxCallable,), {
                "arity": lambda self: arity,
                "call": call,
                "__str__": lambda self: "<native fn>",
            })()

        self.globals.define("clock", make_native_function(
            "Clock", 0, lambda _1, _2, _3: time.time()))

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

    def resolve(self, expr, depth):
        self.locals[expr] = depth

    def execute_block(self, statements, environment):
        previous = self.environment
        try:
            self.environment = environment
            for statement in statements:
                self.execute(statement)
        finally:
            self.environment = previous

    def visit_block_stmt(self, stmt):
        self.execute_block(stmt.statements, Environment(self.environment))

    def visit_class_stmt(self, stmt):
        superclass = None
        if stmt.superclass:
            superclass = self.evaluate(stmt.superclass)
            if not isinstance(superclass, LoxClass):
                raise Interpreter.Error(
                    stmt.superclass.name, "Superclass must be a class.")

        self.environment.define(stmt.name.lexeme, None)

        if superclass:
            self.environment = Environment(self.environment)
            self.environment.define("super", superclass)

        methods = {
            method.name.lexeme: LoxFunction(
                method, self.environment, method.name.lexeme == "init")
            for method in stmt.methods}

        klass = LoxClass(stmt.name.lexeme, superclass, methods)

        if superclass:
            self.environment = self.environment.enclosing

        self.environment.assign(stmt.name, klass)

    def visit_expression_stmt(self, stmt):
        self.evaluate(stmt.expression)

    def visit_function_stmt(self, stmt):
        func = LoxFunction(stmt, self.environment, False)
        self.environment.define(stmt.name.lexeme, func)

    def visit_if_stmt(self, stmt):
        if self.is_truthy(self.evaluate(stmt.condition)):
            self.execute(stmt.then_branch)
        elif stmt.else_branch:
            self.execute(stmt.else_branch)

    def visit_print_stmt(self, stmt):
        value = self.evaluate(stmt.expression)
        print(self.stringify(value))

    def visit_return_stmt(self, stmt):
        value = None
        if stmt.value:
            value = self.evaluate(stmt.value)
        raise Interpreter.Return(value)

    def visit_var_stmt(self, stmt):
        value = None
        if stmt.initializer:
            value = self.evaluate(stmt.initializer)
        self.environment.define(stmt.name.lexeme, value)

    def visit_while_stmt(self, stmt):
        while self.is_truthy(self.evaluate(stmt.condition)):
            self.execute(stmt.body)

    def visit_assign_expr(self, expr):
        value = self.evaluate(expr.value)
        if distance := self.locals.get():
            self.environment.assign_at(distance, expr.name.lexeme, value)
        else:
            self.globals.assign(expr.name)
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

    def visit_call_expr(self, expr):
        callee = self.evaluate(expr.callee)
        if not isinstance(callee, LoxCallable):
            raise Interpreter.Error(
                expr.paren, "Can only call functions and classes.")
        arguments = list(map(self.evaluate, expr.arguments))
        if len(arguments) != callee.arity():
            raise Interpreter.Error(
                expr.paren, f"Expected {callee.arity()} arguments but got {len(arguments)}.")
        return callee.call(self, arguments)

    def visit_get_expr(self, expr):
        obj = self.evaluate(expr.object)
        if isinstance(obj, LoxInstance):
            return obj.get(expr.name)
        raise Interpreter.Error(
            expr.name, "Only instances have properties.")

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

    def visit_set_expr(self, expr):
        obj = self.evaluate(expr.object)
        if not isinstance(obj, LoxInstance):
            raise Interpreter.Error(
                expr.name, "Only instances have fields.")
        value = self.evaluate(expr.value)
        obj.set(expr.name, value)
        return value

    def visit_super_expr(self, expr):
        distance = self.locals.get(expr)
        superclass = self.environment.get_at(distance, "super")
        instance = self.environment.get_at(distance - 1, "this")

        method = superclass.find_method(expr.method.lexeme)

        if not method:
            raise Interpreter.Error(
                expr.method, f"Undefined property '{expr.method.lexeme}'.")

        return method.bind(instance)

    def visit_this_expr(self, expr):
        return self.lookup_variable(expr.keyword, expr)

    def visit_unary_expr(self, expr):
        right = self.evaluate(expr.right)
        match expr.operator.type:
            case "BANG": return not self.is_truthy(right)
            case "MINUS":
                self.check_operands(expr.operator, float, right)
                return -right
            case _: return None

    def visit_variable_expr(self, expr):
        return self.lookup_variable(expr.name, expr)

    def lookup_variable(self, name, expr):
        if expr in self.locals:
            return self.environment.get_at(self.locals[expr], name.lexeme)
        return self.globals.get(name)

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

        resolver = Resolver(PyLox.interpreter)
        for statement in statements:
            resolver.resolve(statement)

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
