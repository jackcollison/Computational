using Parameters, Plots, Printf, JLD2, TableView, DelimitedFiles

include("Function_CK.jl") #import the functions that solve our growth model

prim, res = Initialize()
#@elapsed Solve_model(prim, res)
