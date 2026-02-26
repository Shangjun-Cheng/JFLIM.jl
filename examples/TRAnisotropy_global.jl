import Pkg
Pkg.activate(raw"D:\JuliaProgramm\JFLIM")
Pkg.instantiate()

using JFLIM
using CSV, DataFrames
using Plots

function main()
    # helper: scalar/array -> Vector{Float32}
    asvec(x) = x isa AbstractArray ? vec(Float32.(collect(x))) : Float32[x]

    # 1) load CSV (NO zero truncation)
    csv_path = raw"C:\Users\CHENG\Desktop\PaperSingleFiber\ref\5_decays_anisotropy.csv"
    df = CSV.read(csv_path, DataFrame)

    y1 = Float32.(df[!, "ch1_decay"])
    y2 = Float32.(df[!, "ch2_decay"])

    T = min(length(y1), length(y2))
    y1 = y1[1:T]
    y2 = y2[1:T]

    # 2) put "two channels" into X dimension:
    # data layout: (X, Y, Z, T, C) = (2, 1, 1, T, 1)
    to_fit = Array{Float32}(undef, 2, 1, 1, T, 1)
    to_fit[1,1,1,:,1] = y1   # X=1 -> ch1
    to_fit[2,1,1,:,1] = y2   # X=2 -> ch2

    @show size(to_fit), maximum(y1), maximum(y2)

    # 3) fit (biexp). NOTE: global_tau may or may not share across X in your JFLIM version.
    use_cuda = false
    amp_positive = true

    res0, fwd0 = flim_fit(
        to_fit, ScaleMaximum();
        use_cuda=use_cuda, verbose=true,
        stat=loss_gaussian, iterations=30, num_exponents=2,
        fixed_tau=true, global_tau=true, amp_positive=amp_positive,
        off_start=0.0001f0, bgnoise=1f0
    )

    res, fwd = flim_fit(
        to_fit, ScaleMaximum();
        use_cuda=use_cuda, verbose=true,
        stat=loss_anscombe_pos, iterations=80, num_exponents=2,
        fixed_tau=false, global_tau=true, fixed_offset=false, amp_positive=amp_positive,
        all_start=res0, bgnoise=1f0
    )

    # 4) print results (重点看 τ 的形状是否含 X=2)
    println("\n================ FIT RESULTS ================")
    if haskey(res, :τs)
        println("size(res[:τs]) = ", size(res[:τs]))
        println("res[:τs] = ", res[:τs])
        println("taus as vector = ", asvec(res[:τs]))
    else
        println("No :τs field found. keys(res) = ", collect(keys(res)))
    end

    if haskey(res, :amps)
        println("\nsize(res[:amps]) = ", size(res[:amps]))
    end

    if haskey(res, :offset)
        println("\noffset = ", res[:offset])
    elseif haskey(res, :offsets)
        println("\noffsets = ", res[:offsets])
    end
    println("============================================\n")

    # 5) plot
    y1_data = collect(@view to_fit[1,1,1,:,1])
    y1_fit  = collect(@view fwd[1,1,1,:,1])
    y2_data = collect(@view to_fit[2,1,1,:,1])
    y2_fit  = collect(@view fwd[2,1,1,:,1])

    t = 0:(T-1)

    p1  = plot(t, y1_data, label="X=1 (ch1) data", lw=2)
    plot!(p1, t, y1_fit,  label="X=1 (ch1) fit",  lw=2, ls=:dash)
    p1r = plot(t, y1_data .- y1_fit, label="X=1 residual", lw=2)
    hline!(p1r, [0], label="0")

    p2  = plot(t, y2_data, label="X=2 (ch2) data", lw=2)
    plot!(p2, t, y2_fit,  label="X=2 (ch2) fit",  lw=2, ls=:dash)
    p2r = plot(t, y2_data .- y2_fit, label="X=2 residual", lw=2)
    hline!(p2r, [0], label="0")

    display(plot(p1, p1r, p2, p2r, layout=(4,1), size=(900,1000)))
end

main()