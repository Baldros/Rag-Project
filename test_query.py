from processing.query import query_base
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--consulta", help="Digite sua consulta")
args = parser.parse_args()

if args.consulta:
    consulta = args.consulta
else:
    consulta = input("Digite sua consulta: ")

print("\nResultados da consulta:\n")
query_base(consulta)