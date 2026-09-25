using Revise
using LinearAlgebra
using Profile
using ProfileView
using CSV
using BenchmarkTools
using Debugger

includet("src/jot/stage1.jl")
Revise.track(Stage1, "src/jot/utils.jl")
using .Stage1

f = CSV.File(open("data/example.csv"), header=false).Column1
f = f[1:256]
params = Dict("γ1" => 0.05, "γ2" => 1000.0, "γ3" => 0.05, "β" => 12.5, "a" => 50.0, "κ" => 1e-7)

# Time profiling
dh = DataHolder(f, params);
sl = ADMMSolver(length(f), dh, 300);
@time solve_stage1!(sl);

# Allocation profiling
Profile.Allocs.clear()
dh = DataHolder(f, params);
sl = ADMMSolver(length(f), dh, 300);
Profile.Allocs.@profile solve_stage1!(sl);

PProf.Allocs.pprof(from_c = false)

visualize(sl)
