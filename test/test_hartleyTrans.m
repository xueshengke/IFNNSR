clear
close all

image_bic_batch_2 = hartleyTrans(image_bic_batch_ht, 'i');
norm(image_bic_batch(:)-image_bic_batch_2(:))

image_res_batch_ht2 = hartleyTrans(image_res_batch, 't');
norm(image_res_batch_ht2(:)-image_res_batch_ht(:))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hei = 180;
wid = 240;
x = (hei-1 : -1 : 0)' / hei; % row
y = (0 : wid-1)' / wid; % col
ex = exp(x.^2);
ey = exp(y.^2);
em = ex * ey';
figure; imagesc(em); colorbar

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bic = image_bic_batch(:,:,5);
bic_ht = hartleyTrans(bic, 't');
bic_rec = hartleyTrans(bic_ht, 'i');
norm(bic-bic_rec)

M = size(bic,1);
N = size(bic,2);
bic_fft = 1/sqrt(M*N) * fft2(bic);
bic_fft2 = 1/sqrt(M*N) * dftmtx(M) * bic * dftmtx(N);
norm(bic_fft-bic_fft2)

bic_ht1 = real(bic_fft) - imag(bic_fft);
bic_ht2 = dhtmtx(M) * bic * dhtmtx(N);
norm(bic_ht1 - bic_ht2)
bic_2 = dhtmtx(M) * bic_ht2 * dhtmtx(N);
norm(bic-bic_2)

bic_ht3 = dhtmtx(M) * bic * dhtmtx(N)';
bic_ht4 = dhtmtx(M)' * bic * dhtmtx(N);

norm(bic_ht2 - bic_ht3)
norm(bic_ht3 - bic_ht4)
norm(bic_ht2 - bic_ht4)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F_M = 1/sqrt(M) * dftmtx(M);
F_Mr = real(F_M); F_Mi = imag(F_M);
F2H = F_Mr - F_Mi;
H_M = dhtmtx(M);
norm(F2H-H_M)

F_N = 1/sqrt(N) * dftmtx(N);
F_Nr = real(F_N); F_Ni = imag(F_N);
F2N = F_Nr - F_Ni;
H_N = dhtmtx(N);
norm(F2N-H_N)

bic_ht_test1 = F_Mr * bic * F_Nr - F_Mi * bic * F_Ni ...
            - F_Mi * bic * F_Nr - F_Mr * bic * F_Ni;
norm(bic_ht1-bic_ht_test1)
bic_ht_test2 = F_Mr * bic * F_Nr + F_Mi * bic * F_Ni ...
            - F_Mi * bic * F_Nr - F_Mr * bic * F_Ni;
norm(bic_ht2-bic_ht_test2)

norm(bic_ht_test1-bic_ht_test2)