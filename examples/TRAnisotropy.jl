import Pkg
Pkg.activate(raw"D:\JuliaProgramm\JFLIM")
Pkg.instantiate()
# using DelimitedFiles
using JFLIM
using View5D
using TestImages
using NDTools
using Noise

using CSV
using DataFrames
using Statistics



function simulate_flim_data(img1, img2, ntimes, tau1, tau2)
    tau1 = Float32(tau1)
    tau2 = Float32(tau2)
    times = reorient(0:ntimes-1, Val(4))
    dat = img1 .* exp.(.-times./tau1) .+ img2 .* exp.(.-times./tau2)
    return dat
end

function main()
    #ISM = nothing, img1 = nothing, img2 = nothing, to_fit=nothing
    ISM = nothing
    img1 = nothing
    img2 = nothing
    to_fit = nothing

    if false
        ISM = readdlm("ISMdata.txt")
        ISM = reshape(ISM, (512, 512, 256))
        ISM = ISM[:, :, 50:120]

        tmp = reshape(ISM, (size(ISM,1), size(ISM,2), 1, size(ISM,3), 1))
        to_fit = max.(0f0, Float32.(tmp))
    else

        csv_path = raw"C:\Users\xi49nol\Desktop\LukasFLIM\Archiv\results tr anisotropy\5_decays_anisotropy.csv"
        df = CSV.read(csv_path, DataFrame)

        y1 = Float32.(df[!, "ch1_decay"])
        y2 = Float32.(df[!, "ch2_decay"])

        # 用总强度找到第一个 >0 的点，保证两通道裁剪对齐
        ysum = y1 .+ y2
        idx0 = findfirst(>(0f0), ysum)
        if idx0 !== nothing && idx0 > 1
            y1 = y1[idx0:end]
            y2 = y2[idx0:end]
        end

        T = length(y1)
        to_fit_2ch = Array{Float32}(undef, 1, 1, 1, T, 2)
        to_fit_2ch[1,1,1,:,1] = y1
        to_fit_2ch[1,1,1,:,2] = y2

        @show size(to_fit_2ch), maximum(y1), maximum(y2)

    end

    to_fit_ch1 = to_fit_2ch[:,:,:,:,1:1]
    to_fit_ch2 = to_fit_2ch[:,:,:,:,2:2]

    use_cuda = true
    amp_positive = true

    # ---- ch1 ----
    res1, fwd1 = flim_fit(to_fit_ch1, ScaleMaximum(); use_cuda=use_cuda, verbose=true,
        stat=loss_gaussian, iterations=30, num_exponents=2,
        fixed_tau=true, global_tau=true, amp_positive=amp_positive,
        off_start=0.0001f0, bgnoise=1f0)

    res2_1, fwd2_1 = flim_fit(to_fit_ch1, ScaleMaximum(); use_cuda=use_cuda, verbose=true,
        stat=loss_anscombe_pos, iterations=80, num_exponents=2,
        fixed_tau=false, global_tau=true, fixed_offset=false, amp_positive=amp_positive,
        all_start=res1, bgnoise=1f0)

    y1    = vec(to_fit_ch1[1,1,1,:,1])
    y1fit = vec(fwd2_1[1,1,1,:,1])

    # ---- ch2 ----
    res2, fwd2 = flim_fit(to_fit_ch2, ScaleMaximum(); use_cuda=use_cuda, verbose=true,
        stat=loss_gaussian, iterations=30, num_exponents=2,
        fixed_tau=true, global_tau=true, amp_positive=amp_positive,
        off_start=0.0001f0, bgnoise=1f0)

    res2_2, fwd2_2 = flim_fit(to_fit_ch2, ScaleMaximum(); use_cuda=use_cuda, verbose=true,
        stat=loss_anscombe_pos, iterations=80, num_exponents=2,
        fixed_tau=false, global_tau=true, fixed_offset=false, amp_positive=amp_positive,
        all_start=res2, bgnoise=1f0)

    y2    = vec(to_fit_ch2[1,1,1,:,1])
    y2fit = vec(fwd2_2[1,1,1,:,1])

    @show res2_1[:τs]
    @show res2_2[:τs]

    using Plots

    t = 0:length(y1)-1

    p1 = plot(t, y1, label="ch1 data", lw=2)
    plot!(p1, t, y1fit, label="ch1 fit", lw=2, ls=:dash)
    p1r = plot(t, y1 .- y1fit, label="ch1 residual", lw=2); hline!(p1r, [0], label="0")

    p2 = plot(t, y2, label="ch2 data", lw=2)
    plot!(p2, t, y2fit, label="ch2 fit", lw=2, ls=:dash)
    p2r = plot(t, y2 .- y2fit, label="ch2 residual", lw=2); hline!(p2r, [0], label="0")

    plot(p1, p1r, p2, p2r, layout=(4,1), size=(900,1000))


end
