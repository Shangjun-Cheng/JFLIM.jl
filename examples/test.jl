import Pkg
Pkg.activate(raw"D:\JuliaProgramm\JFLIM")
Pkg.instantiate()

using JFLIM
using CSV, DataFrames
using Plots

function main()
    # -------- 1) 读两通道 decay（不做任何“零点截断”）--------
    csv_path = raw"C:\Users\CHENG\Desktop\PaperSingleFiber\ref\5_decays_anisotropy.csv"
    df = CSV.read(csv_path, DataFrame)

    y1 = Float32.(df[!, "ch1_decay"])
    y2 = Float32.(df[!, "ch2_decay"])

    T1, T2 = length(y1), length(y2)
    T = min(T1, T2)                 # 若两列长度不一致，取共同长度，避免越界
    y1 = y1[1:T]
    y2 = y2[1:T]

# (X,Y,Z,T,C) = (1,1,1,T,2)，把 ch1/ch2 放到 C 维
to_fit = Array{Float32}(undef, 1, 1, 1, T, 2)
to_fit[1,1,1,:,1] = y1   # C=1 -> ch1
to_fit[1,1,1,:,2] = y2   # C=2 -> ch2

    @show size(to_fit_z), maximum(y1), maximum(y2)

    # -------- 3) 两步拟合：先粗拟合固定 τ，再放开 τ（两条曲线共享 τ）--------
    use_cuda = false
    amp_positive = true

    res0, fwd0 = flim_fit(
        to_fit_z, ScaleMaximum();
        use_cuda=use_cuda, verbose=true,
        stat=loss_gaussian, iterations=30, num_exponents=2,
        fixed_tau=true, global_tau=true, amp_positive=amp_positive,
        off_start=0.0001f0, bgnoise=1f0
    )

    res, fwd = flim_fit(
        to_fit_z, ScaleMaximum();
        use_cuda=use_cuda, verbose=true,
        stat=loss_anscombe_pos, iterations=80, num_exponents=2,
        fixed_tau=false, global_tau=true, fixed_offset=false, amp_positive=amp_positive,
        all_start=res0, bgnoise=1f0
    )

    # ====== 打印拟合结果（τ、幅度 amps、offset）======
    println("\n================ FIT RESULTS ================")

    # τs：全局共享寿命（num_exponents=2 => 两个 τ）
    println("taus (shared):")
    @show res[:τs]

    # amps：每个通道（Z=1/2）各自的幅度
    println("\namplitudes (per Z slice):")
    @show size(res[:amps])
    println("amps for ch1 (z=1):")
    @show res[:amps][1,1,1,1,:]
    println("amps for ch2 (z=2):")
    @show res[:amps][1,1,2,1,:]

    # offset：如果有的话（不同版本/设置字段名可能是 :offset 或 :offsets）
    println("\noffset:")
    if haskey(res, :offset)
        @show res[:offset]
    elseif haskey(res, :offsets)
        @show res[:offsets]
    else
        println("No offset field found in res keys = ", collect(keys(res)))
    end

    println("============================================\n")

    @show res[:τs]

    # -------- 4) 取出拟合结果并画图（用 view+collect，避免 vec(::Float32)）--------
    y1_data  = collect(@view to_fit_z[1,1,1,:,1])
    y1_fit   = collect(@view fwd[1,1,1,:,1])
    y2_data  = collect(@view to_fit_z[1,1,2,:,1])
    y2_fit   = collect(@view fwd[1,1,2,:,1])

    t = 0:(T-1)

    p1  = plot(t, y1_data, label="ch1 data", lw=2)
    plot!(p1, t, y1_fit,  label="ch1 fit",  lw=2, ls=:dash)
    p1r = plot(t, y1_data .- y1_fit, label="ch1 residual", lw=2)
    hline!(p1r, [0], label="0")

    p2  = plot(t, y2_data, label="ch2 data", lw=2)
    plot!(p2, t, y2_fit,  label="ch2 fit",  lw=2, ls=:dash)
    p2r = plot(t, y2_data .- y2_fit, label="ch2 residual", lw=2)
    hline!(p2r, [0], label="0")

    display(plot(p1, p1r, p2, p2r, layout=(4,1), size=(900,1000)))
end

main()