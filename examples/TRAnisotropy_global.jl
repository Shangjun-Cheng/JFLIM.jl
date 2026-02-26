import Pkg
Pkg.activate(raw"D:\JuliaProgramm\JFLIM")
Pkg.instantiate()

using JFLIM
using CSV
using DataFrames
using Plots
using Statistics

function load_two_channel_decay(csv_path)
    df = CSV.read(csv_path, DataFrame)

    y1 = Float32.(df[!, "ch1_decay"])
    y2 = Float32.(df[!, "ch2_decay"])

    ysum = y1 .+ y2
    idx0 = findfirst(>(0f0), ysum)
    if idx0 !== nothing && idx0 > 1
        y1 = y1[idx0:end]
        y2 = y2[idx0:end]
    end

    return y1, y2
end

function make_global_input(y1, y2)
    @assert length(y1) == length(y2)

    tcount = length(y1)
    to_fit = Array{Float32}(undef, 2, 1, 1, tcount)
    to_fit[1, 1, 1, :] = y1
    to_fit[2, 1, 1, :] = y2

    return to_fit
end

function estimate_channel_offsets(to_fit; tail_points=40)
    tcount = size(to_fit, 4)
    tstart = max(1, tcount - tail_points + 1)
    tail = @view to_fit[:, :, :, tstart:tcount]
    off_start = Float32.(mean(tail, dims=4))
    return max.(off_start, 1f-4)
end

function tau_key(res)
    for k in keys(res)
        if String(k) == "\u03c4s"
            return k
        end
    end
    error("Could not find tau field in fit result.")
end

function fit_two_channel_global(to_fit; use_cuda=false, amp_positive=true)
    # Channel-specific offset starts from the tail baseline of each channel.
    off_start = estimate_channel_offsets(to_fit)

    res1, _ = flim_fit(
        to_fit,
        ScaleMaximum();
        use_cuda=use_cuda,
        verbose=true,
        stat=loss_gaussian,
        iterations=30,
        num_exponents=2,
        fixed_tau=true,
        global_tau=true,
        fixed_offset=true,
        amp_positive=amp_positive,
        off_start=off_start,
        bgnoise=1f0,
    )

    # Keep taus globally shared by reusing the global tau shape from stage 1
    # while allowing per-channel amplitudes and offsets in stage 2.
    tau_start = res1[tau_key(res1)]
    amp_start = res1[:amps]

    res2, fit2 = flim_fit(
        to_fit,
        ScaleMaximum();
        use_cuda=use_cuda,
        verbose=true,
        stat=loss_anscombe_pos,
        iterations=80,
        num_exponents=2,
        fixed_tau=false,
        global_tau=true,
        fixed_offset=false,
        amp_positive=amp_positive,
        tau_start=tau_start,
        amp_start=amp_start,
        off_start=off_start,
        bgnoise=1f0,
    )

    return res2, fit2
end

function main()
    csv_path = raw"C:\Users\CHENG\Desktop\PaperSingleFiber\ref\5_decays_anisotropy.csv"

    y1, y2 = load_two_channel_decay(csv_path)
    to_fit = make_global_input(y1, y2)

    @show size(to_fit)
    @show maximum(y1), maximum(y2)

    res2, fit2 = fit_two_channel_global(to_fit; use_cuda=false, amp_positive=true)

    y1_fit = vec(fit2[1, 1, 1, :])
    y2_fit = vec(fit2[2, 1, 1, :])

    @show keys(res2)
    @show res2

    t = 0:length(y1)-1

    p1 = plot(t, y1, label="ch1 data", lw=2)
    plot!(p1, t, y1_fit, label="ch1 global fit", lw=2, ls=:dash)
    p1r = plot(t, y1 .- y1_fit, label="ch1 residual", lw=2)
    hline!(p1r, [0], label="0")

    p2 = plot(t, y2, label="ch2 data", lw=2)
    plot!(p2, t, y2_fit, label="ch2 global fit", lw=2, ls=:dash)
    p2r = plot(t, y2 .- y2_fit, label="ch2 residual", lw=2)
    hline!(p2r, [0], label="0")

    display(plot(p1, p1r, p2, p2r, layout=(4, 1), size=(900, 1000)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
