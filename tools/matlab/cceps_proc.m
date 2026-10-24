function [error, x_ccep, ndelay, x_frames_r] = cceps_proc(x_frames, fft_m)
if nargin < 2
    fft_m = 1;
end

[n_frames, n_fft] = size(x_frames);
x_ccep = zeros(n_frames, n_fft);
ndelay = zeros(n_frames, 1);
for i = 1:n_frames
    [x_hat, nd] = cceps(x_frames(i, :), n_fft*fft_m);
    x_ccep(i, :) = x_hat;
    ndelay(i, :) = nd;
end

x_frames_r = zeros(n_frames, n_fft);
for i = 1:n_frames
    x = icceps(x_ccep(i, :), ndelay(i, :));
    x_frames_r(i, :) = x(1:n_fft);
end
error = mean((x_frames - x_frames_r).^2, 'all');

end
