# dice - A maubot plugin that rolls dice.
# Copyright (C) 2019 Tulir Asokan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import Match, Union, Any, Type, Optional
import operator
import random
import math
import ast
import re

from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper
from mautrix.types import EventType, TextMessageEventContent, MessageType, Format
from maubot import Plugin, MessageEvent
from maubot.handlers import command, event

pattern_regex = re.compile("([0-9]{0,9})[dD]([0-9]{1,9})")

_OP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Invert: operator.inv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.RShift: operator.rshift,
    ast.LShift: operator.lshift,
}

_NUM_MAX = 1_000_000_000_000_000
_NUM_MIN = -_NUM_MAX

_OP_LIMITS = {
    ast.Pow: (1000, 1000),
    ast.LShift: (1000, 1000),
    ast.Mult: (1_000_000_000_000_000, 1_000_000_000_000_000),
    ast.Div: (1_000_000_000_000_000, 1_000_000_000_000_000),
    ast.FloorDiv: (1_000_000_000_000_000, 1_000_000_000_000_000),
    ast.Mod: (1_000_000_000_000_000, 1_000_000_000_000_000),
}

_ALLOWED_FUNCS = ["ceil", "copysign", "fabs", "factorial", "gcd", "remainder", "trunc",
                  "exp", "log", "log1p", "log2", "log10", "sqrt",
                  "acos", "asin", "atan", "atan2", "cos", "hypot", "sin", "tan",
                  "degrees", "radians",
                  "acosh", "asinh", "atanh", "cosh", "sinh", "tanh",
                  "erf", "erfc", "gamma", "lgamma"]

_FUNC_MAP = {
    **{func: getattr(math, func) for func in _ALLOWED_FUNCS if hasattr(math, func)},
    "round": round,
    "hash": hash,
    "max": max,
    "min": min,
    "float": float,
    "int": int,
    "abs": abs,
}

_FUNC_LIMITS = {
    "factorial": 1000,
    "exp": 709,
    "sqrt": 1_000_000_000_000_000,
}

_ARG_COUNT_LIMIT = 5


# AST-based calculator from https://stackoverflow.com/a/33030616/2120293
class Calc(ast.NodeVisitor):
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        try:
            left_max, right_max = _OP_LIMITS[op_type]
            if left > left_max or right > right_max:
                raise ValueError(f"Value over bounds in operator {op_type.__name__}")
        except KeyError:
            pass
        try:
            op = _OP_MAP[op_type]
        except KeyError:
            raise SyntaxError(f"Operator {op_type.__name__} not allowed")
        return op(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        try:
            op = _OP_MAP[type(node.op)]
        except KeyError:
            raise SyntaxError(f"Operator {type(node.op).__name__} not allowed")
        return op(operand)

    def visit_Num(self, node: ast.Num) -> Any:
        if node.n > _NUM_MAX or node.n < _NUM_MIN:
            raise ValueError(f"Number out of bounds")
        return node.n

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            if node.value > _NUM_MAX or node.value < _NUM_MIN:
                raise ValueError(f"Number out of bounds")
            return node.n
        return None

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id == "pi":
            return math.pi
        elif node.id == "tau":
            return math.tau
        elif node.id == "e":
            return math.e

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            if node.func.id == "ord" and len(node.args) == 1 and isinstance(node.args[0], ast.Str):
                return ord(node.args[0].s)
            try:
                func = _FUNC_MAP[node.func.id]
            except KeyError:
                raise NameError(f"Function {node.func.id} is not defined")
            args = [self.visit(arg) for arg in node.args]
            kwargs = {kwarg.arg: self.visit(kwarg.value) for kwarg in node.keywords}
            if len(args) + len(kwargs) > _ARG_COUNT_LIMIT:
                raise ValueError("Too many arguments")
            try:
                limit = _FUNC_LIMITS[node.func.id]
                for value in args:
                    if value > limit:
                        raise ValueError(f"Value over bounds for function {node.func.id}")
                for value in kwargs.values():
                    if value > limit:
                        raise ValueError(f"Value over bounds for function {node.func.id}")
            except KeyError:
                pass
            return func(*args, **kwargs)
        raise SyntaxError("Indirect call")

    def visit_Expr(self, node: ast.Expr) -> Any:
        return self.visit(node.value)

    @classmethod
    def evaluate(cls, expression: str) -> Union[int, float]:
        tree = ast.parse(expression)
        return cls().visit(tree.body[0])


class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper) -> None:
        helper.copy("show_statement")
        helper.copy("show_rolls")
        helper.copy("show_rolls_limit")
        helper.copy("gauss_limit")
        helper.copy("result_max_length")
        helper.copy("round_decimals")
        helper.copy("allow_reaction_reroll")


class DiceBot(Plugin):
    show_rolls: bool = False
    show_statement: bool = False
    show_rolls_limit: int = 20
    gauss_limit: int = 100
    result_max_length: int = 512
    round_decimals: int = 2
    allow_reaction_reroll: bool = True

    async def start(self) -> None:
        self.on_external_config_update()

    def on_external_config_update(self) -> None:
        self.config.load_and_update()
        self.show_statement = self.config["show_statement"]
        self.show_rolls = self.config["show_rolls"]
        self.show_rolls_limit = self.config["show_rolls_limit"]
        self.gauss_limit = self.config["gauss_limit"]
        self.result_max_length = self.config["result_max_length"]
        self.round_decimals = self.config["round_decimals"]
        self.allow_reaction_reroll = self.config["allow_reaction_reroll"]

    @classmethod
    def get_config_class(cls) -> Type[Config]:
        return Config

    def _do_roll(self, pattern: str) -> Optional[str]:
        if not pattern:
            return str(random.randint(1, 6))
        if len(pattern) > 64:
            return None

        individual_rolls = [] if self.show_rolls else None

        def randomize(number: int, size: int) -> int:
            if size < 0 or number < 0:
                raise ValueError("randomize() only accepts non-negative values")
            if size == 0 or number == 0:
                return 0
            elif size == 1:
                return number
            _result = 0
            if number < self.gauss_limit:
                individual = [] if self.show_rolls and number < self.show_rolls_limit else None
                for i in range(number):
                    roll = random.randint(1, size)
                    if individual is not None:
                        individual.append(roll)
                    _result += roll
                if individual:
                    individual_rolls.append((number, size, individual))
            else:
                mean = number * (size + 1) / 2
                variance = number * (size ** 2 - 1) / 12
                while _result < number or _result > number * size:
                    _result = int(random.gauss(mean, math.sqrt(variance)))
            return _result

        def replacer(match: Match) -> str:
            number = int(match.group(1) or "1")
            size = int(match.group(2))
            return str(randomize(number, size))

        pattern = pattern_regex.sub(replacer, pattern)
        try:
            result = Calc.evaluate(pattern)
            if self.round_decimals >= 0:
                result = round(result, self.round_decimals)
            result = str(result)
            if len(result) > self.result_max_length:
                raise ValueError("Result too long")
        except (TypeError, NameError, ValueError, SyntaxError, KeyError, OverflowError,
                ZeroDivisionError):
            self.log.debug(f"Failed to evaluate `{pattern}`", exc_info=True)
            return None
        if self.show_statement and pattern != result:
            result = f"{pattern} = {result}"
        if individual_rolls:
            result += "\n\n"
            result += "\n".join(f"{number}d{size}: {' '.join(str(result) for result in results)}  "
                                for number, size, results in individual_rolls)
        return result

    @command.new("roll")
    @command.argument("pattern", pass_raw=True, required=False)
    async def roll(self, evt: MessageEvent, pattern: str) -> None:
        if pattern:
            self.log.debug(f"Handling `{pattern}` from {evt.sender}")
        result = self._do_roll(pattern)
        if result is None:
            await evt.reply("Bad pattern 3:<")
        else:
            await evt.reply(result)

    # The event.on() decorator is not correctly typed and doesn't understand
    # this is a bound method.
    @event.on(EventType.REACTION)  # type: ignore[arg-type]
    async def handle_reaction(self, evt: MessageEvent) -> None:
        """Handle any reaction on a bot roll result to trigger a reroll.

        Flow: reaction → bot's reply (the roll result) → original !roll command.
        The bot's reply was sent via evt.reply(), so it has an in_reply_to
        reference back to the user's !roll message. We follow that chain to
        extract the dice pattern and reroll.
        """
        if not self.allow_reaction_reroll:
            return

        # Ignore our own reactions
        if evt.sender == self.client.mxid:
            return

        # Get the reaction details and fetch the message that was reacted to
        reaction = evt.content.relates_to
        message_event = await self.client.get_event(evt.room_id, reaction.event_id)

        # Only handle reactions to our own messages (the bot's roll results)
        if message_event.sender != self.client.mxid:
            return

        # The reacted message is the bot's reply. Follow the reply chain
        # back to the original !roll command.
        try:
            original_event_id = message_event.content.relates_to.in_reply_to.event_id
        except (AttributeError, KeyError):
            return

        original_event = await self.client.get_event(evt.room_id, original_event_id)

        # Extract the dice pattern from the original !roll command
        body = original_event.content.body.strip()
        roll_match = re.match(r'^!roll\s*(.*)', body, re.IGNORECASE)
        if not roll_match:
            return
        pattern = roll_match.group(1).strip()

        self.log.debug(f"Reaction reroll of `{pattern}` for {evt.sender}")
        result = self._do_roll(pattern)
        if result is None:
            return

        member = await self.client.get_state_event(evt.room_id, EventType.ROOM_MEMBER, evt.sender)
        display_name = member.displayname or evt.sender
        await self.client.send_notice(evt.room_id, text=f"{display_name} rolled: {result}")
