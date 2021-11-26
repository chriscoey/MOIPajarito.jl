# all tests

import Test
import MathOptInterface
const MOI = MathOptInterface
import GLPK
import ECOS

include("MOI_wrapper.jl")
println("starting MOI tests")
Test.@testset "MOI tests" begin
    glpk_opt = MOI.OptimizerWithAttributes(
        GLPK.Optimizer,
        "msg_lev" => 2,
        "tol_int" => 1e-9,
        "tol_bnd" => 1e-8,
        "mip_gap" => 0.0,
    )
    ecos_opt = MOI.OptimizerWithAttributes(ECOS.Optimizer, MOI.Silent() => true)
    TestMOIWrapper.runtests(glpk_opt, ecos_opt)
end
