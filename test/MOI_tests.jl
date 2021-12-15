# MOI tests

module TestMOI

using Test
import MathOptInterface
const MOI = MathOptInterface
import MOIPajarito

function runtests(oa_solver, conic_solver)
    @testset "iterative method" begin
        run_moi_tests(true, oa_solver, conic_solver)
    end
    @testset "one tree method" begin
        run_moi_tests(false, oa_solver, conic_solver)
    end
    return
end

function run_moi_tests(use_iter::Bool, oa_solver, conic_solver)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            MOIPajarito.Optimizer(),
        ),
        Float64,
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("use_iterative_method"), use_iter)
    MOI.set(model, MOI.RawOptimizerAttribute("oa_solver"), oa_solver)
    MOI.set(model, MOI.RawOptimizerAttribute("conic_solver"), conic_solver)
    # MOI.set(model, MOI.RawOptimizerAttribute("time_limit"), 60)
    MOI.set(model, MOI.RawOptimizerAttribute("iteration_limit"), 100)

    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            Float64,
            atol = 2e-4,
            rtol = 2e-4,
            exclude = Any[
                MOI.ConstraintDual,
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.DualObjectiveValue,
                MOI.SolverVersion,
            ],
        ),
        # include = String[],
        exclude = String[
            # TODO(odow): unexpected failure, probably in the bridge layer
            "test_model_UpperBoundAlreadySet",
            "test_model_LowerBoundAlreadySet",
        ],
    )
    return
end

end
