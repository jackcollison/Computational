# PS7: Simulated Method of Moments (SMM)

using Plots, Tables, DataFrames, CSV

include("Model.jl");

D_seed = 1500;
R_seed = 200;

# Using different moments
@elapsed res_4 = estimate(1:2; D_seed = D_seed, R_seed = R_seed)
@elapsed res_5 = estimate(2:3; D_seed = D_seed, R_seed = R_seed)
@elapsed res_6 = estimate(1:3; D_seed = D_seed, R_seed = R_seed)

true_data = Initialize_True_Data(; seed = D_seed);

# Plot true data
plot(true_data.x₀, legend = false, ylabel = L"x_{t}", xlabel= L"t", color = "black")
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/AR1.pdf")

# J_TH surface for first-stage
plot_3d_J_TH(res_4, true_data, 1)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_1_1st.pdf")
plot_3d_J_TH(res_5, true_data, 1)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_2_1st.pdf")
plot_3d_J_TH(res_6, true_data, 1)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_3_1st.pdf")

# J_TH surface for second-stage
plot_3d_J_TH(res_4, true_data, 2)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_1_2st.pdf")
plot_3d_J_TH(res_5, true_data, 2)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_2_2st.pdf")
plot_3d_J_TH(res_6, true_data, 2)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_3_2st.pdf")
