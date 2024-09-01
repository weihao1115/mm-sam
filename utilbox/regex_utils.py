"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.09
"""
import re

# return all the smallest sub-strings surrounded by a pair of angle brackets '<>'
# e.g.
# <exp_root> -> <exp_root>
# <exp<root> -> <root>
# <exp>root> -> <exp>
regex_angle_bracket = re.compile(r"<[^<>]*>")

# return the smallest sub-string surrounded by a pair of square brackets '[]'
# e.g.
# a,b,c,[d,e,[f,g,[h,i,j,k]]] -> [h,i,j,k]
# d,e,[f,g,[h,i,j,k]] -> [h,i,j,k]
# f,g,[h,i,j,k] -> [h,i,j,k]
regex_square_bracket = re.compile(r"\[[^\[\]]*\]")
# return the largest sub-string surrounded by a pair of square brackets '[]'
# e.g.
# a,b,c,[d,e,[f,g,[h,i,j,k]]] -> [d,e,[f,g,[h,i,j,k]]]
# d,e,[f,g,[h,i,j,k]] -> [f,g,[h,i,j,k]]
# f,g,[h,i,j,k] -> [h,i,j,k]
regex_square_bracket_large = re.compile(r"\[.*\]")

# return all the smallest sub-string surrounded by a pair of braces '{}'
# e.g.
# a,b,c,{d,e,{f,g,{h,i,j,k}}} -> {h,i,j,k}
# d,e,{f,g,{h,i,j,k}} -> {h,i,j,k}
# f,g,{h,i,j,k} -> {h,i,j,k}
regex_brace = re.compile(r"{[^{}]*}")

def has_nested_structure(string):
    pattern = r"\[.*\[.*\].*\]"
    match = re.search(pattern, string)
    return bool(match)
