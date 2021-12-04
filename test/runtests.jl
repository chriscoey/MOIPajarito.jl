# all tests

import Test
import MathOptInterface
const MOI = MathOptInterface

import GLPK
oa_solver = MOI.OptimizerWithAttributes(
    GLPK.Optimizer,
    "msg_lev" => 0,
    "tol_int" => 1e-9,
    "tol_bnd" => 1e-8,
    "mip_gap" => 0.0,
)

import ECOS
conic_solver = MOI.OptimizerWithAttributes(ECOS.Optimizer, MOI.Silent() => true)
# import Hypatia
# conic_solver = MOI.OptimizerWithAttributes(Hypatia.Optimizer, MOI.Silent() => true)

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
;