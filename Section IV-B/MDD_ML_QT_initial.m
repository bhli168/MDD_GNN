function [p_initial,P_L_W,variPi_DL,variPi_UL,Psi_DL,Psi_UL] = MDD_ML_QT_initial(Num_MS,Num_Sumsubcarrier,Num_DLsubcarrier,Num_ULsubcarrier,Num_AP,Chi_DL,Chi_UL,Power_AP_W,Power_MS_W,Omega,...
    Upsilon,SI_cap_AP,SI_cap_MS,No,Beta_MS_MS,Beta_AP_AP,IAI_cap_AP,IMI_cap_MS)

cvx_solver sdpt3
cvx_begin quiet

variable p(Num_MS,Num_Sumsubcarrier,Num_AP)
variable a_DL(Num_MS,Num_DLsubcarrier)
variable a_UL(Num_MS,Num_ULsubcarrier)
variable variPi_DL(Num_MS,Num_DLsubcarrier)
variable variPi_UL(Num_MS,Num_ULsubcarrier)
variable Psi_DL(Num_MS,Num_DLsubcarrier)
variable Psi_UL(Num_MS,Num_ULsubcarrier)
expression p_temp(Num_MS,Num_Sumsubcarrier,Num_AP)
p_temp(:,1:Num_DLsubcarrier,:) = p(:,1:Num_DLsubcarrier,:);
p_temp(:,Num_DLsubcarrier+1:end,1) = p(:,Num_DLsubcarrier+1:end,1);
expression P_L_W(Num_AP,1);
for ll = 1:Num_AP
    P_L_W(ll,1) = sum(sum(p_temp(:,1:Num_DLsubcarrier,ll)));
end

maximize sum(sum(a_DL)) + sum(sum(a_UL))
subject to

for ll = 1:Num_AP
    sum(sum(p_temp(:,1:Num_DLsubcarrier,ll))) <= Power_AP_W;
end
for d = 1:Num_MS
    sum(sum(p_temp(d,Num_DLsubcarrier+1:end,1))) <= Power_MS_W;
    
end

for d = 1:Num_MS
    for m = 1:Num_DLsubcarrier
        idxx = find(Omega(:,d,m)>0);
        0 <= variPi_DL(d,m) <= Omega(idxx,d,m).' *  sqrt(squeeze(p_temp(d,m,:)));%% sqrt function ignores the nonzero items
        Psi_DL(d,m) >= SI_cap_MS*(sum(sum(p_temp(d,Num_DLsubcarrier+1:end,1)))) + IMI_cap_MS*sum(sum(Beta_MS_MS(d,:)*p_temp(:,Num_DLsubcarrier+1:end,1)))/Num_Sumsubcarrier + No;
        2 * variPi_DL(d,m) - Psi_DL(d,m) >= exp(Chi_DL/Num_DLsubcarrier) -1 + a_DL(d,m);
        a_DL(d,m) <= 0;
    end
end


for d = 1:Num_MS
    for m = 1:Num_ULsubcarrier      
        0 <= variPi_UL(d,m) <= p_temp(d,Num_DLsubcarrier+m,1) * (Num_AP^2);
        Psi_UL(d,m) >= SI_cap_AP*Upsilon(:,d,m).' * P_L_W...
            + (IAI_cap_AP/Num_Sumsubcarrier)*Upsilon(:,d,m).'* Beta_AP_AP * P_L_W...
            + No * sum(Upsilon(:,d,m));
        2 * 10^-2.5 * sqrt(variPi_UL(d,m)) - 10^-5 * Psi_UL(d,m) >= exp(Chi_UL/Num_ULsubcarrier) - 1 + a_UL(d,m);
        a_UL(d,m) <= 0;
    end
    
end

cvx_end
p_initial = p;

end