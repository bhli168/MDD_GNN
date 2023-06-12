function [R_sum_final,R_sum_MS] = MDD_SumRateCalculate_Cluster(p,Omega,Upsilon,SI_cap_MS,SI_cap_AP,No,Num_AP,Num_MS,Num_DLsubcarrier,Num_ULsubcarrier,Num_Sumsubcarrier,...
    Beta_AP_AP,Beta_MS_MS,IAI_cap_AP,IMI_cap_MS,UL_inter,UL_gain,DL_mui)
P_L_W = zeros(Num_AP,1);
R_sum = zeros(Num_MS,Num_Sumsubcarrier);
A_dm = zeros(Num_MS,Num_DLsubcarrier);
B_dm = zeros(Num_MS,Num_DLsubcarrier);
A_dmbar = zeros(Num_MS,Num_ULsubcarrier);
B_dmbar = zeros(Num_MS,Num_ULsubcarrier);
for ll = 1:Num_AP
    P_L_W(ll,1) = sum(sum(p(:,1:Num_DLsubcarrier,ll)));
end
for ll = 1:Num_AP
    P_L_m(ll,:) = sum(p(:,1:Num_DLsubcarrier,ll));
end
for d = 1:Num_MS
    for m = 1:Num_DLsubcarrier
        A_dm(d,m) = (Omega(:,d,m).' *  sqrt(reshape(p(d,m,:),[Num_AP,1])))^2;
        B_dm(d,m) = SI_cap_MS*(sum(sum(p(d,Num_DLsubcarrier+1:end,1)))) + ...
            IMI_cap_MS*sum(sum(Beta_MS_MS(d,:)*p(:,Num_DLsubcarrier+1:end,1)))/Num_Sumsubcarrier ...
            + No + DL_mui(:,d,m).' * P_L_m(:,m);
        R_sum(d,m) = log(1 + A_dm(d,m)/ B_dm(d,m)) / Num_Sumsubcarrier;
    end
    for m = 1:Num_ULsubcarrier
        A_dmbar(d,m) = p(d,Num_DLsubcarrier+m,1) * (UL_gain(d,m)^2);
        B_dmbar(d,m) = SI_cap_AP*Upsilon(:,d,m).' * P_L_W...
            + (IAI_cap_AP/Num_Sumsubcarrier)*Upsilon(:,d,m).'* Beta_AP_AP * P_L_W...
            + No * sum(Upsilon(:,d,m)) +  UL_inter(d,:,m) * p(:,Num_DLsubcarrier+m,1);
        if B_dmbar(d,m) == 0
            R_sum(d,m + Num_DLsubcarrier) = 0;
        else
            
            R_sum(d,m + Num_DLsubcarrier) = log(1 + A_dmbar(d,m)/ B_dmbar(d,m)) / Num_Sumsubcarrier;
        end
    end
end
R_sum_MS = sum(R_sum,2).';
R_sum_final = sum(R_sum_MS,2);
end