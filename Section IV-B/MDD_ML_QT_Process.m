function [R_sum,p,P_L_W,variPi_DL,variPi_UL,Psi_DL,Psi_UL] = MDD_ML_QT_Process(Num_MS,Num_Sumsubcarrier,Num_DLsubcarrier,Num_ULsubcarrier,Num_AP,Chi_DL,Chi_UL,Power_AP_W,Power_MS_W,Omega,...
    Upsilon,SI_cap_AP,SI_cap_MS,No,Beta_MS_MS,Beta_AP_AP,variPi_DL_iter,variPi_UL_iter,Psi_DL_iter,Psi_UL_iter,z_DL,z_UL,IAI_cap_AP,IMI_cap_MS)

cvx_solver Mosek
cvx_begin quiet
variable p(Num_MS,Num_Sumsubcarrier,Num_AP)
variable tau_DL
variable tau_UL
variable variPi_DL(Num_MS,Num_DLsubcarrier)
variable variPi_UL(Num_MS,Num_ULsubcarrier)
variable Psi_DL(Num_MS,Num_DLsubcarrier)
variable Psi_UL(Num_MS,Num_ULsubcarrier)

expression p_temp(Num_MS,Num_Sumsubcarrier,Num_AP)
p_temp(:,1:Num_DLsubcarrier,:) = p(:,1:Num_DLsubcarrier,:);
p_temp(:,Num_DLsubcarrier+1:end,1) = p(:,Num_DLsubcarrier+1:end,1);

expression P_L_W(Num_AP,1)
for ll = 1:Num_AP
    P_L_W(ll,1) = sum(sum(p_temp(:,1:Num_DLsubcarrier,ll)));
end


expression R_max(Num_MS,Num_Sumsubcarrier)
expression A(Num_MS,Num_Sumsubcarrier)
expression B(Num_MS,Num_Sumsubcarrier)
for d = 1:Num_MS
    for m = 1:Num_DLsubcarrier
        idxx = find(Omega(:,d,m)>0);
        A(d,m) = Omega(idxx,d,m).' *  sqrt(squeeze(p_temp(d,m,:)));
        B(d,m) = SI_cap_MS*(sum(sum(p_temp(d,Num_DLsubcarrier+1:end,1)))) + IMI_cap_MS*sum(sum(Beta_MS_MS(d,:)*p_temp(:,Num_DLsubcarrier+1:end,1)))/Num_Sumsubcarrier + No;
        R_max(d,m) = log(1 + 2 * z_DL(d,m) * A(d,m) - z_DL(d,m)^2 * B(d,m)) / Num_Sumsubcarrier;
    end
    for m = 1:Num_ULsubcarrier
        A(d,m + Num_DLsubcarrier) = p_temp(d,Num_DLsubcarrier+m,1) * (Num_AP^2);
        B(d,m + Num_DLsubcarrier) = SI_cap_AP*Upsilon(:,d,m).' * P_L_W...
        + (IAI_cap_AP/Num_Sumsubcarrier)*Upsilon(:,d,m).'* Beta_AP_AP * P_L_W...
        + No * sum(Upsilon(:,d,m));
        R_max(d,m + Num_DLsubcarrier) = log(1 + 2 * z_UL(d,m) * sqrt(A(d,m + Num_DLsubcarrier)) - z_UL(d,m)^2 * B(d,m + Num_DLsubcarrier)) / Num_Sumsubcarrier;
    end
end

    
maximize sum(sum(R_max))
subject to


for ll = 1:Num_AP
    sum(sum(p_temp(:,1:Num_DLsubcarrier,ll))) <= Power_AP_W;
end
for d = 1:Num_MS
    sum(sum(p_temp(d,Num_DLsubcarrier+1:end,1))) <= Power_MS_W;
    
end

for d = 1:Num_MS
    for m = 1:Num_DLsubcarrier
        0 <= variPi_DL(d,m) <= A(d,m);
        Psi_DL(d,m) >= B(d,m);
        (2 * variPi_DL_iter(d,m) / Psi_DL_iter(d,m)) * variPi_DL(d,m) - (variPi_DL_iter(d,m)^2 / Psi_DL_iter(d,m)^2) * Psi_DL(d,m) >= exp(Chi_DL/Num_DLsubcarrier)-1;
    end
end

for d = 1:Num_MS
    for m = 1:Num_ULsubcarrier
        0 <= variPi_UL(d,m) <= A(d,m + Num_DLsubcarrier);
        Psi_UL(d,m) >= B(d,m + Num_DLsubcarrier);
        (2 * sqrt(variPi_UL_iter(d,m)) / Psi_UL_iter(d,m)) * sqrt(variPi_UL(d,m)) - (variPi_UL_iter(d,m) / Psi_UL_iter(d,m)^2) * Psi_UL(d,m) >= exp(Chi_UL/Num_ULsubcarrier)-1;
    end
    
end

cvx_end

R_sum = sum(sum(R_max));

end