import re


# ============================================================
# Tokenisation du code Solidity
# ============================================================

TOKEN_PATTERN = re.compile(
    r"""
    //.*?$                              | # commentaires ligne
    /\*.*?\*/                           | # commentaires multilignes
    "(?:\\.|[^"\\])*"                   | # chaînes entre guillemets
    '(?:\\.|[^'\\])*'                   | # caractères entre apostrophes

    \b0x[0-9a-fA-F]+\b                  | # nombres hexadécimaux

    \b\d+(?:\.\d+)+\b                   | # versions / nombres décimaux
    \b\d+\b                             | # nombres entiers

    [A-Za-z_][A-Za-z0-9_]*              | # identifiants / mots-clés

    ==|!=|<=|>=|&&|\|\||\+\+|--|=>      | # opérateurs composés

    [{}()\[\];,.:+\-*/%=<>!&|^~?]         # symboles Solidity
    """,
    re.VERBOSE | re.MULTILINE | re.DOTALL
)


def tokenize_solidity(code):
    """
    Transforme un code Solidity en une liste de tokens.
    """

    if not isinstance(code, str):
        return []

    tokens = []

    for match in TOKEN_PATTERN.finditer(code):

        token = match.group(0).strip()

        if token:
            tokens.append(token)

    return tokens


# ============================================================
# Test du tokenizeur
# ============================================================

if __name__ == "__main__":

    example = """
    pragma solidity ^0.4.25;

    contract Wallet {

        uint balance;

        function deposit() public payable {

            balance += msg.value;

        }
    }
    """

    tokens = tokenize_solidity(example)

    print("Tokens :")
    print(tokens)

    print()
    print("Nombre de tokens :", len(tokens))