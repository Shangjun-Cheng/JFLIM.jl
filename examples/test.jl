import Pkg
Pkg.activate(raw"D:\JuliaProgramm\JFLIM")
Pkg.instantiate()

using JFLIM
using CSV, DataFrames
using Plots

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
    t_start = 0          # inclusive, 0-based index
    t_end   = 639          # inclusive, 0-based index
    @assert 0 ≤ t_start ≤ t_end < T

    idx = (t_start+1):(t_end+1)   # Julia is 1-based
    Tfit = length(idx)

    # build to_fit using only the window
    # =========================
    # 2) put "two channels" into X dimension
    # data layout: (X, Y, Z, T, C) = (2, 1, 1, T, 1)
    # =========================
    to_fit = Array{Float32}(undef, 2, 1, 1, Tfit, 1)
    to_fit[1,1,1,:,1] = y1[idx]
    to_fit[2,1,1,:,1] = y2[idx]

    @show size(to_fit), maximum(y1), maximum(y2)

    # =========================
    # 3) fit settings
    # =========================
    use_cuda = false
    amp_positive = true

    # =========================
    # 4) first fit: fixed taus + gaussian noise
    # =========================
    res, fwd = flim_fit(
        to_fit, ScaleMaximum();
        use_cuda=use_cuda, verbose=true,
        stat=loss_gaussian, iterations=30, num_exponents=2,
        fixed_tau=true, global_tau=true, amp_positive=amp_positive,
        # tau_start=reorient([10f0, 20f0], Val(5)),
        off_start=0.0001f0, bgnoise=1f0
    )

    # =========================
    # 5) second fit: free global taus + anscombe noise
    # reuse previous result as starting point
    # =========================
    res2, fwd2 = flim_fit(
        to_fit, ScaleMaximum();
        use_cuda=use_cuda, verbose=true,
        stat=loss_anscombe_pos, iterations=80, num_exponents=2,
        fixed_tau=false, global_tau=true, fixed_offset=false, amp_positive=amp_positive,
        all_start=res, bgnoise=1f0
    )

    # =========================
    # 6.5) derive explicit bi-exp equations in PHYSICAL TIME (ns)
    # =========================

    # ---- build physical time axis (ns) for the fitted window ----
    # Prefer: CSV first column (time). If not available, fall back to dt = 0.0293 ns/bin.
    dt_ns = 0.0293f0  # fallback, if time column not found

    function _get_time_axis_ns(df, idx, dt_ns)
        # try common column names first
        for name in ("time", "t", "Time", "T", "ns", "time_ns", "Time (ns)")
            if name in names(df)
                tcol = Float32.(df[!, name])
                return collect(@view tcol[idx])
            end
        end
        # otherwise: assume first column is time
        try
            tcol = Float32.(df[!, 1])
            return collect(@view tcol[idx])
        catch
            # final fallback: uniform time grid
            Tfit_local = length(idx)
            return dt_ns .* Float32.(0:(Tfit_local-1))
        end
    end

    t_ns = _get_time_axis_ns(df, idx, dt_ns)
    Tfit = length(t_ns)

    # ---- extract fitted params ----
    taus = vec(res2[:τs])             # could be in "bins" or already in "ns" depending on JFLIM
    b    = haskey(res2, :offset) ? res2[:offset] : res2[:offsets]  # be tolerant

    # ---- decide whether taus are in bins or ns ----
    # Heuristic: if taus are too large/small relative to your time axis, convert from bins -> ns.
    # You can override by forcing convert_tau_bins_to_ns = true/false.
    convert_tau_bins_to_ns = false
    if maximum(t_ns) > 0.5f0  # you have a meaningful ns axis
        # if taus are O(10^2~10^3) while times are O(10), they are probably in bins
        if maximum(taus) > 50f0 && maximum(t_ns) < 50f0
            convert_tau_bins_to_ns = true
        end
    end

    taus_ns = convert_tau_bins_to_ns ? (taus .* dt_ns) : taus

    println("\n================ FIT RESULTS (TIME UNITS) ================")
    println("Assumed dt_ns (fallback) = ", dt_ns, " ns/bin")
    println("convert_tau_bins_to_ns = ", convert_tau_bins_to_ns)
    println("taus (raw) = ", taus)
    println("taus (ns ) = ", taus_ns)
    println("offset b   = ", b)
    println("=========================================================\n")

    # ---- choose two time points (ns) for solving amplitudes ----
    # use indices 1 and 11 (0 and 10 bins) if possible, but take their PHYSICAL times
    i1 = 1
    i2 = min(11, Tfit)      # 11 => bin 10 (0-based), consistent with your old choice

    t1 = t_ns[i1]
    t2 = t_ns[i2]

    function amps_from_two_points_time(y_t1, y_t2, b, τ1_ns, τ2_ns, t1_ns, t2_ns)
        e11 = exp(-t1_ns/τ1_ns); e12 = exp(-t1_ns/τ2_ns)
        e21 = exp(-t2_ns/τ1_ns); e22 = exp(-t2_ns/τ2_ns)
        r1 = y_t1 - b
        r2 = y_t2 - b
        det = e11*e22 - e12*e21
        A1 = ( r1*e22 - e12*r2) / det
        A2 = (-r1*e21 + e11*r2) / det
        return A1, A2
    end

    # fitted values from forward model at those two indices
    y1_t1 = fwd2[1,1,1,i1,1];  y1_t2 = fwd2[1,1,1,i2,1]
    y2_t1 = fwd2[2,1,1,i1,1];  y2_t2 = fwd2[2,1,1,i2,1]

    A11, A12 = amps_from_two_points_time(y1_t1, y1_t2, b, taus_ns[1], taus_ns[2], t1, t2)
    A21, A22 = amps_from_two_points_time(y2_t1, y2_t2, b, taus_ns[1], taus_ns[2], t1, t2)

    println("\n================ FINAL FIT EQUATIONS (t in ns) ================")
    println("Curve 1 (ch1): y1(t) = $(b) + $(A11) * exp(-t/$(taus_ns[1])) + $(A12) * exp(-t/$(taus_ns[2]))")
    println("Curve 2 (ch2): y2(t) = $(b) + $(A21) * exp(-t/$(taus_ns[1])) + $(A22) * exp(-t/$(taus_ns[2]))")
    println("================================================================\n")

    # =========================
    # 7) plot (data vs fit + residuals) in PHYSICAL TIME (ns)
    # =========================
    y1_data = collect(@view to_fit[1,1,1,:,1])
    y1_fit  = collect(@view fwd2[1,1,1,:,1])
    y2_data = collect(@view to_fit[2,1,1,:,1])
    y2_fit  = collect(@view fwd2[2,1,1,:,1])

    @assert length(y1_data) == Tfit
    @assert length(y1_fit)  == Tfit
    @assert length(y2_data) == Tfit
    @assert length(y2_fit)  == Tfit

    # also plot an "analytic bi-exp" curve using the solved amplitudes (should be perfectly smooth)
    y1_model = b .+ A11 .* exp.(-t_ns ./ taus_ns[1]) .+ A12 .* exp.(-t_ns ./ taus_ns[2])
    y2_model = b .+ A21 .* exp.(-t_ns ./ taus_ns[1]) .+ A22 .* exp.(-t_ns ./ taus_ns[2])

    try
        closeall()
    catch
    end

    p1  = plot(t_ns, y1_data, label="ch1 data", lw=2, xlabel="time (ns)", ylabel="counts")
    plot!(p1, t_ns, y1_fit,   label="ch1 fwd2 fit", lw=2, ls=:dash)
    plot!(p1, t_ns, y1_model, label="ch1 analytic bi-exp", lw=2, ls=:dot)

    p1r = plot(t_ns, y1_data .- y1_fit, label="ch1 residual (data - fwd2)", lw=2, xlabel="time (ns)", ylabel="residual")
    hline!(p1r, [0], label="0")

    p2  = plot(t_ns, y2_data, label="ch2 data", lw=2, xlabel="time (ns)", ylabel="counts")
    plot!(p2, t_ns, y2_fit,   label="ch2 fwd2 fit", lw=2, ls=:dash)
    plot!(p2, t_ns, y2_model, label="ch2 analytic bi-exp", lw=2, ls=:dot)

    p2r = plot(t_ns, y2_data .- y2_fit, label="ch2 residual (data - fwd2)", lw=2, xlabel="time (ns)", ylabel="residual")
    hline!(p2r, [0], label="0")

    plt = plot(p1, p1r, p2, p2r, layout=(4,1), size=(900,1000))
    display(plt)
end
main()