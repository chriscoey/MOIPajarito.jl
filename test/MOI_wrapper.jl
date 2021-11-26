# MOI tests

module TestMOIWrapper

import Test
import MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities
import MOIPajarito

function runtests(oa_solver, conic_solver)
    Test.@testset "use iterative method: $use_iter" for use_iter in (true, false)
        run_moi_tests(use_iter, oa_solver, conic_solver)
    end
    return
end

function run_moi_tests(use_iter::Bool, oa_solver, conic_solver)
    paj_opt = MOIPajarito.Optimizer()
    # MOI.set(paj_opt, MOI.Silent(), true)
    MOI.set(paj_opt, MOI.RawOptimizerAttribute("use_iterative_method"), use_iter)
    MOI.set(paj_opt, MOI.RawOptimizerAttribute("oa_solver"), oa_solver)
    MOI.set(paj_opt, MOI.RawOptimizerAttribute("conic_solver"), conic_solver)

    caching_opt = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(MOIU.Model{Float64}()),
        MOI.Bridges.full_bridge_optimizer(paj_opt, Float64),
    )

    config = MOI.Test.Config(
        atol = 1e-4,
        rtol = 1e-4,
        exclude = Any[
            MOI.ConstraintDual,
            MOI.ConstraintBasisStatus,
            MOI.DualObjectiveValue,
        ],
    )

    excludes = String[
        # not implemented:
        "test_attribute_SolverVersion",
        # invalid model:
        # "test_constraint_ZeroOne_bounds_3",
        "test_linear_VectorAffineFunction_empty_row",
        # CachingOptimizer does not throw if optimizer not attached:
        "test_model_copy_to_UnsupportedAttribute",
        "test_model_copy_to_UnsupportedConstraint",
    ]
    excludes = String[]

    includes = String["test_conic_SecondOrderCone",]
    # includes = String[]

    MOI.Test.runtests(caching_opt, config, exclude = excludes, include = includes)
    return
end

end
