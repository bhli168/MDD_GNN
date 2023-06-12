function [] = MDD_ML_SE_main_Cluster(SquareLength,Num_AP,Num_MS)

%--------------------------------Setup---------------------------------%

rng('shuffle')
%SquareLength = 400; %square 400m * 400m
%Num_AP = 24;
Num_AP_ant = 1;
%Num_MS = 12;
Num_AP_per_MS = 1;
Num_DLsubcarrier = 4;
Num_ULsubcarrier = 2;
Num_Sumsubcarrier = Num_DLsubcarrier + Num_ULsubcarrier;
Power_AP_dBm = 40;
Power_AP_W = (10^(0.1 * Power_AP_dBm) * 10^-3);
Power_MS_dBm = 30;
Power_MS_W = (10^(0.1 * Power_MS_dBm) * 10^-3);

SI_cap_MS = 10^(-11);
SI_cap_AP = 10^(-12);
IMI_cap_MS = 10^(-4.2);
IAI_cap_AP = 10^(-7.2);
Chi_DL = 0; %% lower bound QoS requirements
Chi_UL = 0;

%% Channel Parameters
Num_DelayTaps = 4;
Bandwidth = 100*10^6;
Subcarrier_bandwidth = 15 * 10^3;
SymbolDuration = 66.67 * 10^-6;
GuardPeriod = 1 * SymbolDuration;
Carrier_Frequency = 5 * 10^9;
Wavelength = 3*10^8 / Carrier_Frequency;
No_dB = -174+10*log10(Bandwidth); % Noise power in dBm, -174 dBm/Hz
No = 10^(.1*No_dB) * 10^-3;    % Noise power absolute


filename_1 = ['test_',num2str(Num_AP),num2str(Num_MS),'_clu.mat']; %% train or test
load(filename_1)

ii = 0;
while ii<=999
    ii = ii + 1;
    try
        Beta_MS_MS = squeeze(MM(ii,:,:));
        Beta_AP_AP = squeeze(AA(ii,:,:));
        Omega = squeeze(f_zf(ii,:,:,:));
        Upsilon = squeeze(w_zf(ii,:,:,:));
        DL_mui = squeeze(DL_MUI(ii,:,:,:));
        UL_mui = squeeze(UL_MUI(ii,:,:,:));
        UL_gain = squeeze(UL_Gain(ii,:,:));
        UL_inter = squeeze(UL_Inter(ii,:,:,:));

        %% power initialization
        [p,P_L_W,variPi_DL,variPi_UL,Psi_DL,Psi_UL] = MDD_ML_QT_initial_Cluster(Num_MS,Num_Sumsubcarrier,Num_DLsubcarrier,Num_ULsubcarrier,Num_AP,Chi_DL,Chi_UL,Power_AP_W,...
            Power_MS_W,Omega,Upsilon,DL_mui,UL_inter,UL_gain,SI_cap_AP,SI_cap_MS,No,Beta_MS_MS,Beta_AP_AP,IAI_cap_AP,IMI_cap_MS);
        p_DL = p(:,1:Num_DLsubcarrier,:);
        p_UL = p(:,Num_DLsubcarrier+1:end,1);
        z_DL = zeros(Num_MS,Num_DLsubcarrier);
        z_UL = zeros(Num_MS,Num_ULsubcarrier);
        A_dm = zeros(Num_MS,Num_DLsubcarrier);
        B_dm = zeros(Num_MS,Num_DLsubcarrier);
        A_dmbar = zeros(Num_MS,Num_ULsubcarrier);
        B_dmbar = zeros(Num_MS,Num_ULsubcarrier);

        iter = 0;
        diff = 100;
        p_iter = cell(1,10);
        R = 0;
        while (diff > 0.1 && iter <= 10)
            iter = iter + 1;
            variPi_DL_iter = variPi_DL;
            variPi_UL_iter = variPi_UL;
            Psi_DL_iter = Psi_DL;
            Psi_UL_iter = Psi_UL;
            for ll = 1:Num_AP
                P_L_m(ll,:) = sum(p_DL(:,:,ll));
            end
            % Update z
            for d = 1:Num_MS
                for m = 1:Num_DLsubcarrier
                    A_dm(d,m) = (Omega(:,d,m).' *  sqrt(reshape(p_DL(d,m,:),[Num_AP,1])))^2;
                    B_dm(d,m) = SI_cap_MS*(sum(sum(p_UL(d,:)))) + IMI_cap_MS*sum(sum(Beta_MS_MS(d,:)*p_UL(:,:)))/Num_Sumsubcarrier + No +...
                        DL_mui(:,d,m).' * P_L_m(:,m);
                    z_DL(d,m) = abs(sqrt(A_dm(d,m)) / B_dm(d,m));
                end
                for m = 1:Num_ULsubcarrier
                    A_dmbar(d,m) = p_UL(d,m) * (UL_gain(d,m)^2);
                    B_dmbar(d,m) = SI_cap_AP*Upsilon(:,d,m).' * P_L_W...
                        + (IAI_cap_AP/Num_Sumsubcarrier)*Upsilon(:,d,m).'* Beta_AP_AP * P_L_W...
                        + No * sum(Upsilon(:,d,m)) +  UL_inter(d,:,m) * p_UL(:,m);
                    if B_dmbar(d,m)==0
                        z_UL(d,m) = 0;
                    else
                        z_UL(d,m) = abs(sqrt(A_dmbar(d,m)) / B_dmbar(d,m));
                    end
                end
            end


            [R_sum,p,P_L_W,variPi_DL,variPi_UL,Psi_DL,Psi_UL] = MDD_ML_QT_Process_Cluster(Num_MS,Num_Sumsubcarrier,Num_DLsubcarrier,Num_ULsubcarrier,Num_AP,Chi_DL,Chi_UL,Power_AP_W,Power_MS_W,Omega,...
                Upsilon,DL_mui,UL_inter,UL_gain,SI_cap_AP,SI_cap_MS,No,Beta_MS_MS,Beta_AP_AP,variPi_DL_iter,variPi_UL_iter,Psi_DL_iter,Psi_UL_iter,z_DL,z_UL,IAI_cap_AP,IMI_cap_MS);

            R(iter) = R_sum;
            p_iter{1,iter} = p;
            p_DL = p(:,1:Num_DLsubcarrier,:);
            p_UL = p(:,Num_DLsubcarrier+1:end,1);
            if iter > 1
                diff = R(iter) - R(iter-1);
            end
            if diff < 0 || isnan(diff)
                p_final = p_iter{1,iter-1};
                break;
            else
                p_final = p_iter{1,iter};
            end

        end
        [R_AfW,R_AfW_MS] = MDD_SumRateCalculate_Cluster(p_final,Omega,Upsilon,SI_cap_MS,SI_cap_AP,No,Num_AP,Num_MS,Num_DLsubcarrier,Num_ULsubcarrier,Num_Sumsubcarrier,...
            Beta_AP_AP,Beta_MS_MS,IAI_cap_AP,IMI_cap_MS,UL_inter,UL_gain,DL_mui);

        SE(ii) = R_AfW;
    catch
        SE(ii) = 0;
        save('tem.mat')
        clear all
        load('tem.mat')
        delete('tem.mat')
    end

end
toc
filename_2 = ['GNN_cluster_',num2str(Num_AP),num2str(Num_MS),'.txt']; %% train or test
fid1 = fopen(filename_2,'at+');
fprintf(fid1,'%6.6f ',SE);
fprintf(fid1,'\n');
fclose(fid1);
