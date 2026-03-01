import Pkg
Pkg.activate(raw"D:\JuliaProgramm\JFLIM")
Pkg.instantiate()

using JFLIM
using CSV, DataFrames
using Plots
using View5D
function main()
    # helper: scalar/array -> Vector{Float32}
    # asvec(x) = x isa AbstractArray ? vec(Float32.(collect(x))) : Float32[x]
    csv_path = raw"D:\CodeFromCheng\MDSimu\Dataset\mean_r_curve.csv" # 或 mean_P_curve.csv 
    df = CSV.read(csv_path, DataFrame) 
    y = Float32.(df[!, "decay"]) 
    T = length(y)
    # =========================
    # 1) load CSV (NO zero truncation)
    # =========================

    
    t_start = 0          # inclusive, 0-based index
    t_end   = 100      # inclusive, 0-based index
    @assert 0 ≤ t_start ≤ t_end < T

    # Julia is 1-based
    start1 = t_start + 1
    end1   = t_end + 1

    # 每隔 step 个点取样一次（在窗口 [start1, end1] 内）
    step = 1
    @assert step ≥ 1

    idx_ds = start1:step:end1
    y_ds   = y[idx_ds]

    # Tfit = length(y_ds)

    to_fit = Array{Float32}(undef, 1, 1, 1, T)
    # to_fit[1,1,1,:] = y_ds   
    to_fit[1,1,1,:] = y   

    @show size(to_fit)

    # =========================
    # 3) fit settings
    # =========================
    use_cuda = true
    amp_positive = true
    # first fit with fixed taus and gaussian noise
    res, fwd = flim_fit(to_fit, ScaleMaximum(); use_cuda = use_cuda, verbose=true, stat=loss_gaussian, iterations=30, num_exponents=2,
                        fixed_tau=true, global_tau=true, amp_positive=amp_positive,
                        # tau_start=reorient([10f0, 20f0], Val(5)),
                        off_start=0.0001f0, bgnoise=1f0);
    res[:τs] 

    # ... and then improve the fit with free global taus and poisson or anscombe noise. reuse the previous result as starting point
    res2, fwd2 = flim_fit(to_fit, ScaleMaximum(); use_cuda = use_cuda, verbose=true, stat=loss_anscombe_pos, iterations=80, num_exponents=2,
                        fixed_tau=false, global_tau=true, fixed_offset=false, amp_positive=amp_positive,
                        all_start=res, bgnoise=1f0);

    amps = res2[:amps]
    τs = res2[:τs] 
    res2.offset

    
end

main()