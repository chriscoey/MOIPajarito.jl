#=
exponential cone
(u, v, w) : (v > 0, w ≥ v * exp(u / v)) or (u ≤ 0, v = 0, w ≥ 0)
dual cone is:
(p, q, r) : (p < 0, r ≥ -p * exp(q / p - 1)) or (p = 0, q ≥ 0, r ≥ 0)
=#

function add_init_cuts(opt::Optimizer, s_vars::Vector{VR}, ::MOI.ExponentialCone)
    @assert length(s_vars) == 3
    (u, v, w) = s_vars
    # add variable bounds and some separation cuts at fixed q points with p = -1
    qs = [-4.0, 1.0, 5.0]
    JuMP.@constraints(opt.oa_model, begin
        v >= 0
        w >= 0
        [q in qs], -u + q * v + exp(-q - 1) * w >= 0
    end)
    return 3
end

function add_subp_cuts(
    opt::Optimizer,
    z::Vector{Float64},
    s_vars::Vector{VR},
    ::MOI.ExponentialCone,
)
    (p, q, r) = z
    (u, v, w) = s_vars
    if p > -1e-8
        # p is near zero: extreme cut is (0, q, r)
        @warn("here")
        if min(q, r) > 1e-12
            expr = JuMP.@expression(opt.oa_model, q * v + r * w)
        else
            return 0
        end
    else
        # extreme cut is (p, q, -p * exp(q / p - 1))
        expr = JuMP.@expression(opt.oa_model, p * u + q * v - p * exp(q / p - 1) * w)
    end
    return add_cut(expr, opt)
end

# function add_sep_cuts(
#     opt::Optimizer,
#     s::Vector{Float64},
#     s_vars::Vector{VR},
#     ::MOI.ExponentialCone,
# )
#     # check (p, q, r) ∉ K
#     (p, q, r) = s
#     if min(q, r, r - q * exp(p / q)) > -opt.tol_feas
#         return 0
#     end

#     (u, v, w) = s_vars
#     if p < 1e-7
#         # p is near zero: extreme cut is ...
#         if min(q, r) > 1e-12
#             expr = JuMP.@expression(opt.oa_model, ...)
#         else
#             return 0
#         end
#     else
#         # extreme cut is ...
#         expr = JuMP.@expression(opt.oa_model, )
#     end
#     return add_cut(expr, opt)
# end

# if s_val <= 1e-7
#     # s is (almost) zero: violation is t
#     viol = t_val

#     if add_cuts && (viol > m.mip_feas_tol)
#         # TODO: for now, error if r is too small
#         if r_val <= 1e-12
#             error("Cannot add exp cone separation cut on point ($r_val, $s_val, $t_val)\n")
#         end

#         # K* separation cut on (r,s,t) is (t/r, -2*log(exp(1)*t/2r), -2)
#         u_val = t_val/r_val
#         v_val = -2.0 * (1.0 + log(u_val/2.0))
#         is_viol_cut = add_cut_exp!(m, r, s, t, u_val, v_val, -2.)
#     end
# else
#     # s is significantly positive: violation is s*exp(t/s) - r
#     ets = exp(t_val/s_val)
#     viol = s_val*ets - r_val

#     if add_cuts && (viol > m.mip_feas_tol)
#         # K* separation cut on (r,s,t) is (1, (t-s)/s*exp(t/s), -exp(t/s))
#         v_val = (t_val - s_val)/s_val*ets
#         is_viol_cut = add_cut_exp!(m, r, s, t, 1., v_val, -ets)
#     end
# end
