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

    # =========================
    # 1) load CSV (NO zero truncation)
    # =========================
    csv_path = raw"C:\Users\CHENG\Desktop\PaperSingleFiber\ref\5_decays_anisotropy.csv"
    df = CSV.read(csv_path, DataFrame)

    y1 = Float32.(df[!, "ch1_decay"])
    y2 = Float32.(df[!, "ch2_decay"])

    T = min(length(y1), length(y2))
    y1 = y1[1:T]
    y2 = y2[1:T]

    # =========================
    # choose fit window (manual)
    # =========================
    t_start = 391          # inclusive, 0-based index
    t_end   = 796          # inclusive, 0-based index
    @assert 0 ≤ t_start ≤ t_end < T

    idx = (t_start+1):(t_end+1)   # Julia is 1-based
    Tfit = length(idx)

    # build to_fit using only the window
    # =========================
    # 2) put "two channels" into X dimension
    # data layout: (X, Y, Z, T, C) = (2, 1, 1, T, 1)
    # =========================
    to_fit = Array{Float32}(undef, 2, 1, 1, Tfit)
    to_fit[1,1,1,:] = y1[idx]
    to_fit[2,1,1,:] = y2[idx]

    @show size(to_fit), maximum(y1), maximum(y2)

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

    # @vt to_fit fwd2


    # =========================
    # 4) plot: data (to_fit) vs fit (fwd2)
    # =========================
    t = collect(0:(Tfit-1))  # bin index; if you have dt, multiply here

    # robust extract as 1D vectors
    data_ch1 = vec(to_fit[1,1,1,:])
    data_ch2 = vec(to_fit[2,1,1,:])

    fit_ch1  = ndims(fwd2) == 5 ? vec(fwd2[1,1,1,:,1]) : vec(fwd2[1,1,1,:])
    fit_ch2  = ndims(fwd2) == 5 ? vec(fwd2[2,1,1,:,1]) : vec(fwd2[2,1,1,:])

    p1 = plot(t, data_ch1; label="to_fit ch1", xlabel="time bin", ylabel="counts", title="Channel 1")
    plot!(p1, t, fit_ch1; label="fwd2 ch1")

    p2 = plot(t, data_ch2; label="to_fit ch2", xlabel="time bin", ylabel="counts", title="Channel 2")
    plot!(p2, t, fit_ch2; label="fwd2 ch2")

    p = plot(p1, p2; layout=(2,1), size=(900,700))
    display(p)

    # optional: save
    # savefig(p, "to_fit_vs_fwd2.png")
    
end

main()