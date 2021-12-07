# all tests

import Test
import MathOptInterface
const MOI = MathOptInterface

import GLPK
oa_solver = MOI.OptimizerWithAttributes(
    GLPK.Optimizer,
    MOI.Silent() => true,
    "tol_int" => 1e-9,
    "tol_bnd" => 1e-9,
    "mip_gap" => 1e-9,
)
# import Cbc
# oa_solver = MOI.OptimizerWithAttributes(
#     Cbc.Optimizer,
#     MOI.Silent() => true,
#     "integerTolerance" => 1e-9,
#     "primalTolerance" => 1e-9,
#     "ratioGap" => 1e-9,
# )

import ECOS
conic_solver = MOI.OptimizerWithAttributes(ECOS.Optimizer, MOI.Silent() => true)
# import Hypatia
# conic_solver = MOI.OptimizerWithAttributes(Hypatia.Optimizer, MOI.Silent() => true)

println("starting Pajarito tests")
Test.@testset "Pajarito tests" begin
    println("starting MOI tests")
    include("MOI_tests.jl")
    Test.@testset "MOI tests" begin
        TestMOI.runtests(oa_solver, conic_solver)
    end

    println("starting JuMP tests")
    include("JuMP_tests.jl")
    Test.@testset "JuMP tests" begin
        TestJuMP.runtests(oa_solver, conic_solver)
    end
end
