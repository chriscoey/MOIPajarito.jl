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


# include("MOI_tests.jl")
# println("starting MOI tests")
# Test.@testset "MOI tests" begin
#     TestMOI.runtests(oa_solver, conic_solver)
# end

include("JuMP_tests.jl")
println("starting JuMP tests")
Test.@testset "JuMP tests" begin
    TestJuMP.runtests(oa_solver, conic_solver)
end
