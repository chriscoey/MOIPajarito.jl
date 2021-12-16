# TODO this is temporary
# so I don't need to tag so many versions of Hypatia to get CI running here
# also later remove Hypatia from Project.toml deps (only keep in test deps)
# and remove Pkg from test deps
# using Pkg
# pkg"add Hypatia#master"
# Pkg.instantiate()

# all tests

import Test
import MathOptInterface
const MOI = MathOptInterface

import GLPK
oa_solver = MOI.OptimizerWithAttributes(
    GLPK.Optimizer,
    MOI.Silent() => true,
    "tol_int" => 1e-10,
    "tol_bnd" => 1e-10,
    "mip_gap" => 1e-10,
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
    #     TestMOI.runtests(oa_solver, hypatia)
    # end

    println("starting JuMP tests")
    include("JuMP_tests.jl")
    Test.@testset "JuMP tests" begin
        TestJuMP.runtests(oa_solver, hypatia)
    end

    println("starting CBF tests")
    include("CBF_tests.jl")
    Test.@testset "CBF tests" begin
        TestCBF.runtests(oa_solver, hypatia)
    end
end
