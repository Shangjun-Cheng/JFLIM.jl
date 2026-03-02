import Pkg
Pkg.activate(raw"D:\JuliaProgramm\JFLIM")
Pkg.instantiate()

using JFLIM
using CSV, DataFrames
using Plots
using View5D
function main()
    # helper: scalar/array -> Vector{Float32}
    asvec(x) = x isa AbstractArray ? vec(Float32.(collect(x))) : Float32[x]
    # csv_path = raw"D:\CodeFromCheng\MDSimu\Dataset\mean_r_curve.csv" 
    csv_path = raw"D:\JuliaProgramm\JFLIM\examples\data\mean_r_curve.csv"
    df = CSV.read(csv_path, DataFrame) 
    y = Float32.(df[!, "decay"]) 
    T = length(y)
    # =========================
    # 1) load CSV (single curve, no truncation)
    # =========================
    y = Float32.(df[!, "decay"])
    T = length(y)

    # Fit window (manual, 0-based indices)
    t_start = 0          # inclusive, 0-based index
    t_end   = 1000          # inclusive, 0-based index
    @assert 1 ≤ t_start+1 ≤ t_end +1≤ T
    idx = (t_start+1):(t_end+1)   # Julia is 1-based
    Tfit = length(idx)
    # build to_fit using only the window
    # =========================
    # 2) pack the single curve into JFLIM layout
    # data layout: (X, Y, Z, T, C) = (1, 1, 1, T, 1)
    # =========================
    to_fit = Array{Float32}(undef, 1, 1, 1, Tfit)
    to_fit[1,1,1,:] = y[idx]

    @show size(to_fit), maximum(y)

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