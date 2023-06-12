function [] = MDD_ML_Produce_Cluster(SquareLength,Num_AP,Num_MS)

%--------------------------------Setup---------------------------------%
rng('shuffle')
%SquareLength = 500; %square 400m * 400m
%Num_AP = 38;
Num_AP_ant = 1;
%Num_MS = 19;
Num_DLsubcarrier = 	4;
Num_ULsubcarrier = 2;
Num_Sumsubcarrier = Num_DLsubcarrier + Num_ULsubcarrier;
SI_cap_MS = 10^(-11);
SI_cap_AP = 10^(-12);

%% Channel Parameters
Num_DelayTaps = 4;
%% Channel Modelling
Im = eye(Num_Sumsubcarrier,Num_Sumsubcarrier);
Psih = Im(:,1:Num_DelayTaps);
Subcarrier_index_ran = randperm(Num_Sumsubcarrier); %% Choose index of DL/UL subcarriers
ULsubcarrier_index_ran = Subcarrier_index_ran(Num_DLsubcarrier+1:end);
ULPhi_ran = Im(ULsubcarrier_index_ran,:);
Subcarrier_index = (1:1:Num_Sumsubcarrier);
ULsubcarrier_index = (1:Num_Sumsubcarrier/Num_ULsubcarrier:Num_Sumsubcarrier);
DLsubcarrier_index = setdiff(Subcarrier_index,ULsubcarrier_index);
DLPhi = Im(DLsubcarrier_index,:);
ULPhi = Im(ULsubcarrier_index,:);
F = zeros(Num_Sumsubcarrier,Num_Sumsubcarrier);
for m = 1:Num_Sumsubcarrier
    for n = 1:Num_Sumsubcarrier
        F(m,n) = (1/sqrt(Num_Sumsubcarrier)) * exp(-1i * 2 * pi * (m-1) * (n-1) / Num_Sumsubcarrier); %FFT
    end
end

iteration = 1000;

for ii = 1:iteration
    sha_F = 4;
    [Beta_AP_AP,Beta_AP_MS,Beta_MS_MS, AP_AP_PL, MS_MS_PL, AP_MS_PL, AP_p, MS_p] = PLsetup(Num_AP,Num_MS,SquareLength,sha_F);
    AA(ii,:,:) = Beta_AP_AP;
    MM(ii,:,:) = Beta_MS_MS;
    AM(ii,:,:) = Beta_AP_MS;
    AA_dis(ii,:,:) = AP_AP_PL;
    MM_dis(ii,:,:) = MS_MS_PL;
    AM_dis(ii,:,:) = AP_MS_PL;
    APp(ii,:,:) = AP_p;
    MSp(ii,:,:) = MS_p;

    g_AP_AP = cell(Num_AP,Num_AP);
    g_AP_MS = cell(Num_AP,Num_MS);
    g_MS_MS = cell(Num_MS,Num_MS);

    for m = 1:Num_AP
        for n = 1:Num_AP
            g_AP_AP{m,n} = (sqrt(Beta_AP_AP(m,n)) * sqrt((1/Num_DelayTaps)/2)) .* (randn(Num_DelayTaps,Num_AP_ant,Num_AP_ant) + 1i * randn(Num_DelayTaps,Num_AP_ant,Num_AP_ant));
        end
    end

    for m = 1:Num_AP
        for n = 1:Num_MS
            g_AP_MS{m,n} = (sqrt(Beta_AP_MS(m,n)) * sqrt((1/Num_DelayTaps)/2)) .* (randn(Num_DelayTaps,Num_AP_ant) + 1i * randn(Num_DelayTaps,Num_AP_ant));
        end
    end

    for m = 1:Num_MS
        for n = 1:Num_MS
            g_MS_MS{m,n} = (sqrt(Beta_MS_MS(m,n)) * sqrt((1/Num_DelayTaps)/2)) .* (randn(Num_DelayTaps,1) + 1i * randn(Num_DelayTaps,1));
        end
    end

    H_AP_SI = cell(1,Num_AP);
    for m = 1:Num_AP
        H_AP_SI{1,m} = sqrt(SI_cap_AP/2) .* (randn(Num_AP_ant,Num_AP_ant) + 1i * randn(Num_AP_ant,Num_AP_ant));
    end
    H_MS_SI = cell(1,Num_MS);
    for m = 1:Num_MS
        H_MS_SI{1,m} = sqrt(SI_cap_MS/2) .* (randn + 1i * randn);
    end

    [~,H_AP_MS_DL,~,~,H_AP_MS_UL,~] = MDD_TDCSI_FDCSI(g_AP_AP,g_AP_MS,g_MS_MS,Num_AP,Num_MS,Num_AP_ant,Num_DelayTaps,DLPhi,ULPhi,F,Psih,Num_DLsubcarrier,Num_ULsubcarrier);
    MS_cluster = cell(1,Num_MS);
    AP_cluster = cell(1,Num_AP);
    clu = [];
    clu2 = 1:Num_AP;
    for dd = 1:Num_MS
        MS_cluster{1,dd} = find(min(AP_MS_PL(setdiff(clu2,clu),dd))==AP_MS_PL(:,dd));
        clu = [clu MS_cluster{1,dd}];
        AP_cluster{1,MS_cluster{1,dd}} = dd;
    end
    clu3 = setdiff(clu2,clu);
    for ll = 1:length(clu3)
        temp = find(min(AP_MS_PL(clu3(ll),:))==AP_MS_PL(clu3(ll),:));
        MS_cluster{1,temp} = [MS_cluster{1,temp} clu3(ll)];
        AP_cluster{1,clu3(ll)} = temp;
    end

    H_DL = cell(1,Num_AP);
    H_UL = cell(1,Num_AP);
    for ll = 1:Num_AP
        H_DL{1,ll} = cell2mat(H_AP_MS_DL(ll,:).');
        H_UL{1,ll} = cell2mat(H_AP_MS_UL(ll,:).');
    end



    F_ZF = cell(Num_AP,Num_DLsubcarrier);
    W_ZF = cell(Num_AP,Num_ULsubcarrier);
    Omega = zeros(Num_AP,Num_MS,Num_DLsubcarrier);
    Omega_MUI = zeros(Num_AP,Num_MS,Num_DLsubcarrier);
    Upsilon = zeros(Num_AP,Num_MS,Num_ULsubcarrier);
    Upsilon_MUI = zeros(Num_AP,Num_MS,Num_ULsubcarrier);

    for ll = 1:Num_AP
        for m = 1:Num_DLsubcarrier
            temp = H_DL{1,ll}(AP_cluster{1,ll},:,m);
            F_ZF{ll,m} = temp' * pinv(temp * temp');
            Omega(ll,AP_cluster{1,ll},m) = 1.0 ./ vecnorm(F_ZF{ll,m});
            Omega_MUI(ll,:,m) =  H_DL{1,ll}(:,:,m) * (F_ZF{ll,m}./vecnorm(F_ZF{ll,m}));
            Omega_MUI(ll,:,m) = abs(Omega_MUI(ll,:,m)).^2;
            Omega_MUI(ll,AP_cluster{1,ll},m) = 0;
        end
        for m = 1:Num_ULsubcarrier
            temp = H_UL{1,ll}(AP_cluster{1,ll},:,m);
            W_ZF{ll,m} = (temp.') * pinv((temp.')' * (temp.'));
            Upsilon(ll,AP_cluster{1,ll},m) = vecnorm(W_ZF{ll,m}).^2;
            Upsilon_MUI(ll,:,m) = W_ZF{ll,m}' * H_UL{1,ll}(:,:,m);
            Upsilon_MUI(ll,:,m) = abs(Upsilon_MUI(ll,:,m)).^2;
            Upsilon_MUI(ll,AP_cluster{1,ll},m) = 0;
        end
    end
    UL_gain = zeros(Num_MS,Num_ULsubcarrier);
    UL_inter = zeros(Num_MS,Num_MS,Num_ULsubcarrier);
    for dd = 1:Num_MS
        for m = 1:Num_ULsubcarrier
            tmp = find(Upsilon_MUI(:,dd,m)==0);
            UL_inter(dd,:,m) = sum(Upsilon_MUI(tmp,:,m),1);
            UL_gain(dd,m) = length(tmp);
        end

    end

    f_zf(ii,:,:,:) = Omega;
    w_zf(ii,:,:,:) = Upsilon;
    DL_MUI(ii,:,:,:) = Omega_MUI;
    UL_MUI(ii,:,:,:) = Upsilon_MUI;
    UL_Gain(ii,:,:) = UL_gain;
    UL_Inter(ii,:,:,:) = UL_inter;
end
filename = ['test_',num2str(Num_AP),num2str(Num_MS),'clu.mat']; %% train or test
save(filename, 'f_zf', 'w_zf', 'AA','MM','AM','AA_dis', 'MM_dis','AM_dis','APp', 'MSp','DL_MUI','UL_MUI','UL_Inter','UL_Gain')

end