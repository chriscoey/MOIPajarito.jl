# all tests

import Test
import MathOptInterface
const MOI = MathOptInterface

import GLPK
oa_solver = MOI.OptimizerWithAttributes(
    GLPK.Optimizer,
    MOI.Silent() => true,
    "tol_int" => 1e-10,
    "tol_bnd" => 1e-12,
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
ecos = MOI.OptimizerWithAttributes(ECOS.Optimizer, MOI.Silent() => true)

import Hypatia
hypatia = MOI.OptimizerWithAttributes(
    Hypatia.Optimizer,
    MOI.Silent() => true,
    # "tol_feas" => 1e-8,
    # "tol_rel_opt" => 1e-8,
    # "tol_abs_opt" => 1e-8,
)

println("starting Pajarito tests")
Test.@testset "Pajarito tests" begin
    println("starting MOI tests")
    include("MOI_tests.jl")
    Test.@testset "MOI tests" begin
        TestMOI.runtests(oa_solver, ecos)
    end

    println("starting JuMP tests")
    include("JuMP_tests.jl")
    Test.@testset "JuMP tests" begin
        TestJuMP.runtests(oa_solver, hypatia)
    end
end
