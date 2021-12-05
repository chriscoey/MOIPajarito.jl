# JuMP instance tests

module TestJuMP

using Test
import MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities
import JuMP
import MOIPajarito

function runtests(oa_solver, conic_solver)
    @testset "iterative method" run_jump_tests(true, oa_solver, conic_solver)
    @testset "one tree method" run_jump_tests(false, oa_solver, conic_solver)
    return
end

function run_jump_tests(use_iter::Bool, oa_solver, conic_solver)
    opt = JuMP.optimizer_with_attributes(
        MOIPajarito.Optimizer,
        "verbose" => false,
        "use_iterative_method" => use_iter,
        "oa_solver" => oa_solver,
        "conic_solver" => conic_solver,
        "iteration_limit" => 100,
        "time_limit" => 5,
    )
    test_insts = filter(x -> startswith(string(x), "_"), names(@__MODULE__; all = true))
    @testset "$inst" for inst in test_insts
        getfield(@__MODULE__, inst)(opt)
    end
    return
end

function _soc1(opt)
    TOL = 1e-4
    m = JuMP.Model(opt)

    JuMP.@variable(m, x)
    JuMP.@objective(m, Min, -x)
    xlb1 = JuMP.@constraint(m, x >= 4)
    soc1 = JuMP.@constraint(m, [3.5, x] in JuMP.SecondOrderCone())
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.INFEASIBLE
    @test JuMP.primal_status(m) == MOI.NO_SOLUTION

    JuMP.delete(m, xlb1)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), -3.5, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -3.5, atol = TOL)
    @test isapprox(JuMP.value(x), 3.5, atol = TOL)

    xlb2 = JuMP.@constraint(m, x >= 3.1)
    JuMP.set_integer(x)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.INFEASIBLE

    JuMP.delete(m, xlb2)
    JuMP.@constraint(m, x >= 0.5)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test isapprox(JuMP.objective_value(m), -3, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -3, atol = TOL)
    @test isapprox(JuMP.value(x), 3, atol = TOL)

    JuMP.@objective(m, Max, -3x)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test isapprox(JuMP.objective_value(m), -3, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -3, atol = TOL)
    @test isapprox(JuMP.value(x), 1, atol = TOL)

    return
end

function _soc2(opt)
    TOL = 1e-4
    m = JuMP.Model(opt)

    JuMP.@variable(m, x)
    JuMP.@variable(m, y)
    JuMP.@variable(m, z, Int)
    JuMP.@constraint(m, z <= 2.5)
    JuMP.@objective(m, Min, x + 2y)
    JuMP.@constraint(m, [z, x, y] in JuMP.SecondOrderCone())

    JuMP.set_integer(x)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    opt_obj = -1 - 2 * sqrt(3)
    @test isapprox(JuMP.objective_value(m), opt_obj, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), opt_obj, atol = TOL)
    @test isapprox(JuMP.value(x), -1, atol = TOL)
    @test isapprox(JuMP.value(y), -sqrt(3), atol = TOL)
    @test isapprox(JuMP.value(z), 2, atol = TOL)

    # TODO see https://github.com/jump-dev/MathOptInterface.jl/issues/1698
    JuMP.unset_integer(x)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    opt_obj = -2 * sqrt(5)
    @test isapprox(JuMP.objective_value(m), opt_obj, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), opt_obj, atol = TOL)
    @test isapprox(abs2(JuMP.value(x)) + abs2(JuMP.value(y)), 4, atol = TOL)
    @test isapprox(JuMP.value(z), 2, atol = TOL)

    return
end

function _soc3(opt)
    TOL = 1e-4
    d = 3
    m = JuMP.Model(opt)
    JuMP.@variable(m, x[1:d], Bin)
    JuMP.@constraint(m, vcat(sqrt(d - 1) / 2, x .- 0.5) in JuMP.SecondOrderCone())
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.INFEASIBLE
    @test JuMP.primal_status(m) == MOI.NO_SOLUTION
    return
end

end
