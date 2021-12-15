import Test
import MathOptInterface
const MOI = MathOptInterface

import GLPK
glpk = MOI.OptimizerWithAttributes(
    GLPK.Optimizer,
    MOI.Silent() => true,
    "tol_int" => 1e-10,
    "tol_bnd" => 1e-10,
    "mip_gap" => 1e-10,
)

import Gurobi
gurobi = MOI.OptimizerWithAttributes(
    Gurobi.Optimizer,
    MOI.Silent() => true,
    "IntFeasTol" => 1e-9,
    "FeasibilityTol" => 1e-9,
    "MIPGap" => 1e-9,
)

import Hypatia
hypatia = MOI.OptimizerWithAttributes(
    Hypatia.Optimizer,
    MOI.Silent() => true,
    "near_factor" => 100,
    "tol_feas" => 1e-8,
    "tol_rel_opt" => 1e-8,
    "tol_abs_opt" => 1e-8,
    "tol_illposed" => 1e-8,
    "tol_slow" => 2e-2,
    "tol_inconsistent" => 1e-7,
)

println("starting Pajarito tests")
Test.@testset "Pajarito tests" begin
    # println("starting MOI tests")
    # include("MOI_tests.jl")
    # Test.@testset "MOI tests" begin
    #     TestMOI.runtests(glpk, hypatia)
    #     # TestMOI.runtests(gurobi, hypatia)
    # end

    println("starting JuMP tests")
    include("JuMP_tests.jl")
    Test.@testset "JuMP tests" begin
        # TestJuMP.runtests(glpk, hypatia)
        TestJuMP.runtests(gurobi, hypatia)
    end

    # println("starting CBF tests")
    # include("CBF_tests.jl")
    # Test.@testset "CBF tests" begin
    #     TestCBF.runtests(glpk, hypatia)
    #     # TestCBF.runtests(gurobi, hypatia)
    # end
end
