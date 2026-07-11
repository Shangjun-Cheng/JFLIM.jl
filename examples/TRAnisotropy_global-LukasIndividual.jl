import Pkg
Pkg.activate(raw"D:\JuliaProgramm\JFLIM")
Pkg.instantiate()

using JFLIM
using CSV, DataFrames
using NDTools: reorient
using Plots
using View5D

function load_lukas_csv_decays(csv_path, decay_column)
    isfile(csv_path) || error("CSV file not found: $csv_path")

    df = CSV.read(csv_path, DataFrame)
    column_names = Symbol.(names(df))
    required_columns = [decay_column]
    missing_columns = setdiff(required_columns, column_names)
    isempty(missing_columns) || error("CSV is missing required column(s): $(join(string.(missing_columns), ", "))")

    decay_sum = vec(Float32.(df[!, decay_column]))

    println("csv file path: ", csv_path)
    println("decay column: ", decay_column)
    println("size of decay: ", size(decay_sum))
    println("length of y1: ", length(decay_sum))

    return decay_sum
end

function main()
    # helper: scalar/array -> Vector{Float32}
    asvec(x) = x isa AbstractArray ? vec(Float32.(collect(x))) : Float32[x]

    # =========================
    # 1) load new Lukas summed DY490 csv data (NO zero truncation)
    # =========================
    # New Lukas summed DY490 dataset
    csv_path = raw"D:\JuliaProgramm\JFLIM\examples\data\DY490-full-sum.csv"

    # CSV column:
    # parallel_decay_sum: summed decay channel used for the fit
    parallel_column = :parallel_raw_sum
    perpendicular_column = :perpendicular_raw_sum

    y1 = load_lukas_csv_decays(csv_path, parallel_column)
    # y1 = load_lukas_csv_decays(csv_path, perpendicular_column)

    T = length(y1)

    println("\n========== Single decay input ==========")
    println("Using CSV path: ", csv_path)
    println("Using column: ", parallel_column)
    println("y1 length: ", length(y1))
    println("========================================\n")

    # =========================
    # choose fit window (manual)
    # =========================
    # t_start = 391          # inclusive, 0-based index
    # t_end   = 796          # inclusive, 0-based index
    # @assert 0 ≤ t_start ≤ t_end < T

    t_start = 1          # inclusive, 0-based index
    t_end   = 1100          # inclusive, 0-based index
    @assert 0 ≤ t_start ≤ t_end < T

    idx = (t_start+1):(t_end+1)   # Julia is 1-based
    Tfit = length(idx)

    # build to_fit using only the window
    # =========================
    # 2) pack the single decay into JFLIM layout
    # data layout: (X, Y, Z, T) = (1, 1, 1, T)
    # =========================
    to_fit = Array{Float32}(undef, 1, 1, 1, Tfit)
    to_fit[1,1,1,:] = y1[idx]

    @show size(to_fit), maximum(y1)

    # =========================
    # 3) fit settings
    # =========================
    use_cuda = true
    amp_positive = true
    # first fit with fixed taus and gaussian noise
    res, fwd = flim_fit(to_fit, ScaleMaximum(); use_cuda = use_cuda, verbose=true, stat=loss_gaussian, iterations=30, num_exponents=2,
                        fixed_tau=true, global_tau=true, amp_positive=amp_positive,
                        tau_start=reorient([10f0, 20f0], Val(5)),
                        off_start=0.0001f0, bgnoise=1f0);
    res[:τs] 

    # ... and then improve the fit with free global taus and poisson or anscombe noise. reuse the previous result as starting point
    res2, fwd2 = flim_fit(to_fit, ScaleMaximum(); use_cuda = use_cuda, verbose=true, stat=loss_anscombe_pos, iterations=80, num_exponents=2,
                        fixed_tau=false, global_tau=true, fixed_offset=false, amp_positive=amp_positive,
                        all_start=res, bgnoise=1f0);

    amps = res2[:amps]
    τs = res2[:τs] 
    offset = res2[:offset]

    println("\n========== Fit parameters ==========")
    println("Final fit result object: res2")
    println("Second-stage initial values: all_start = res")
    println("Fit settings: fixed_tau = false, global_tau = true, fixed_offset = false, amp_positive = ", amp_positive)

    println("-- amplitudes --")
    println("size(res2[:amps]) = ", size(res2[:amps]))
    initial_amps = asvec(res[:amps])
    fitted_amps = asvec(res2[:amps])
    for i in eachindex(fitted_amps)
        init_value = i <= length(initial_amps) ? initial_amps[i] : missing
        println("amplitude[$i]: initial value = ", init_value,
                ", lower bound = ", amp_positive ? 0 : "-Inf",
                ", upper bound = Inf, fitted value = ", fitted_amps[i])
    end

    initial_taus = asvec(res[:τs])
    fitted_taus = asvec(res2[:τs])
    println("-- lifetimes --")
    for i in eachindex(fitted_taus)
        init_value = i <= length(initial_taus) ? initial_taus[i] : missing
        println("lifetime_tau[$i]: initial value = ", init_value,
                ", lower bound = 0, upper bound = Inf, fitted value = ", fitted_taus[i])
    end

    initial_offsets = asvec(res[:offset])
    fitted_offsets = asvec(res2[:offset])
    println("-- background / offset --")
    for i in eachindex(fitted_offsets)
        init_value = i <= length(initial_offsets) ? initial_offsets[i] : missing
        println("background_offset[$i]: initial value = ", init_value,
                ", lower bound = 0, upper bound = Inf, fitted value = ", fitted_offsets[i])
    end

    if haskey(res2, :t0)
        initial_t0 = haskey(res, :t0) ? asvec(res.t0) : Float32[]
        fitted_t0 = asvec(res2.t0)
        println("-- IRF shift / t0 --")
        for i in eachindex(fitted_t0)
            init_value = i <= length(initial_t0) ? initial_t0[i] : missing
            println("irf_shift_t0[$i]: initial value = ", init_value,
                    ", lower bound = not specified, upper bound = not specified, fitted value = ", fitted_t0[i])
        end
    end

    println("Residual / chi-square / AIC / BIC / optimizer convergence: not stored in res2 by flim_fit.")
    println("====================================\n")

    # @vt to_fit fwd2


    # =========================
    # 4) plot: data (to_fit) vs fit (fwd2)
    # =========================
    t = collect(0:(Tfit-1))  # bin index; if you have dt, multiply here

    # robust extract as 1D vectors
    data_y1 = vec(to_fit[1,1,1,:])
    fit_y1  = ndims(fwd2) == 5 ? vec(fwd2[1,1,1,:,1]) : vec(fwd2[1,1,1,:])

    p = plot(t, data_y1; label="to_fit y1", xlabel="time bin", ylabel="counts", title="Single decay fit")
    plot!(p, t, fit_y1; label="fwd2 y1")
    display(p)

    # optional: save
    # savefig(p, "to_fit_vs_fwd2.png")
    
end

main()
