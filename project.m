%%% Project
clc
clear
% Generate color palette
% 16 Standard ANSI colors
ansi = [0,0,0; 128,0,0; 0,128,0; 128,128,0; 0,0,128; 128,0,128; 0,128,128; 192,192,192;
        128,128,128; 255,0,0; 0,255,0; 255,255,0; 0,0,255; 255,0,255; 0,255,255; 255,255,255] / 255;

% 216 colors (the 6x6x6 color cube)
vals = [0, 95, 135, 175, 215, 255] / 255;
[R, G, B] = ndgrid(vals, vals, vals);
cube = [R(:), G(:), B(:)];

% 24 Grayscale levels (gradient from dark to light)
grays_val = linspace(8, 238, 24)' / 255;
grays = [grays_val, grays_val, grays_val];

% Combine into the final 256x3 dataset
my_256_palette = [ansi; cube; grays];
my_256_palette = my_256_palette(1:256, :); % Final array

%% Visualize the colors 
% Reshape the 256x3 palette into a 16x16 grid
palette_grid = reshape(my_256_palette, [16, 16, 3]);

% Display the grid
figure;
imshow(palette_grid, 'InitialMagnification', 'fit');
axis on;
title('256 Color Dataset (16x16 Grid)');

xlabel('Column');
ylabel('Row');

%% color test

num_colors = 50;

% Convert the 256 palette to LAB for accurate human-vision math
my_palette_3d = reshape(my_256_palette, [256, 1, 3]);
my_palette_lab = reshape(rgb2lab(my_palette_3d), [256, 3]);

% CORRECTED: Grab the 5th output (midx) which contains the actual row numbers!
[~, ~, ~, ~, medoid_indices] = kmedoids(my_palette_lab, num_colors, 'Distance', 'euclidean');

% Pull your final 50 most representative colors
diverse_50_palette = my_256_palette(medoid_indices, :);

% next 

num_to_pick = 50;

% Convert the 256 palette to LAB
my_palette_3d = reshape(my_256_palette, [256, 1, 3]);
my_palette_lab = reshape(rgb2lab(my_palette_3d), [256, 3]);

selected_idx = zeros(num_to_pick, 1);
selected_idx(1) = 1; % Start with the very first color

% Track the minimum distance to the colors we've already picked
min_distances = inf(256, 1);

for i = 2:num_to_pick
    % Look at the color we just picked
    last_picked_lab = my_palette_lab(selected_idx(i-1), :);
    
    % Measure the distance from that color to ALL 256 colors
    dist_to_last = sqrt(sum((my_palette_lab - last_picked_lab).^2, 2));
    
    % Update our record of how close each color is to our selected group
    min_distances = min(min_distances, dist_to_last);
    
    % Pick the color that has the MAXIMUM minimum distance!
    [~, next_idx] = max(min_distances);
    selected_idx(i) = next_idx;
end

% Pull your final 50 highly-separated colors
max_spread_50_palette = my_256_palette(selected_idx, :);

%% visualize

palette_grid = reshape(diverse_50_palette, [5, 10, 3]);
palette_grid2 = reshape(max_spread_50_palette, [5, 10, 3]);

% Display the grid
figure;
subplot(1,2,1);
imshow(palette_grid, 'InitialMagnification', 'fit');
axis on;
title('diverce 50 Color Dataset (5x10 Grid)');
xlabel('Column');
ylabel('Row');
subplot(1,2,2);
imshow(palette_grid2, 'InitialMagnification', 'fit');
axis on;
title('50 Color Dataset (5x10 Grid)');
xlabel('Column');
ylabel('Row');

%% Load Image and get its main color clusters
clc
img_org = im2double(imread("peppers_color.tif"));
%img_org = im2double(imread("lake.jpg"));
%img_org = im2double(imread("sea_sky.jpg"));
%img_org = im2double(imread("woman_portrait.jpg"));

% Get dimenssions of input image
[rows, cols, channels] = size(img_org);

% set output image hight and width
img_hight = 1080;
img_width = floor((1080 * cols/rows));

% Check if the original image is smaller than the target output (warning message)
if rows < img_hight || cols < img_width
    warning_msg = sprintf(['Warning: Your input image (%dx%d) is smaller than the target mosaic size (%dx%d).\n\n', ...
                           'Because the script uses "nearest" interpolation, stretching a small image will create ', ...
                           'heavy blockiness and artificial jagged edges. This can confuse the shape-rotation ', ...
                           'algorithm and lower the quality of your final mosaic.\n\n', ...
                           'Do you want to continue anyway?'], cols, rows, img_hight, img_width);
    
    % Trigger popup
    user_choice = questdlg(warning_msg, 'Upscale Warning', 'Continue', 'Cancel', 'Continue');
    
    % Abort the script if they don't choose to Continue
    if strcmp(user_choice, 'Cancel') || isempty(user_choice)
        disp('Script aborted: Image too small.');
        return; % Stops the script entirely
    end
end

% resize orgininal image
img_work = imresize(img_org, [img_hight,img_width], "nearest");

% Convert original image to CIELAB
img_lab = rgb2lab(img_work);

% Flatten to Nx3 list of pixels
pixel_data_lab = reshape(img_lab, [], 3);

% Subset Selection (K-means)
num_colors = 64;

% Create a darker palette to find our correct colors
representative_ratio = 0.72; % Rough average ratio of our shapes
darken_palette_rgb = my_256_palette * representative_ratio;
darken_palette_3d = reshape(darken_palette_rgb, [256, 1, 3]);
darken_palette_lab = reshape(rgb2lab(darken_palette_3d), [256, 3]);

% K-Means Clustering
[~, image_centroids_lab] = kmeans(pixel_data_lab, num_colors, 'MaxIter', 300, 'EmptyAction', 'drop');
idx = knnsearch(darken_palette_lab, image_centroids_lab);

unique_idx = unique(idx);
repro_palette = my_256_palette(unique_idx, :); % select the brighter colors so when darken matches original
actual_num_colors = size(repro_palette, 1);

%% Define Shapes and their specific Palettes
clc
bar_12 = zeros(12, 12);
bar_12(3:10, 1:11) = 1;

%square_outlier = zeros(12, 12); 
%square_outlier(2:11, 2:11) = 1;

circle_outlier = [0 0 0 0 0 1 1 0 0 0 0 0;
                  0 0 0 1 1 1 1 1 1 0 0 0;
                  0 0 1 1 1 1 1 1 1 1 0 0;
                  0 1 1 1 1 1 1 1 1 1 1 0;
                  0 1 1 1 1 1 1 1 1 1 1 0;
                  1 1 1 1 1 1 1 1 1 1 1 1;
                  1 1 1 1 1 1 1 1 1 1 1 1;
                  0 1 1 1 1 1 1 1 1 1 1 0;
                  0 1 1 1 1 1 1 1 1 1 1 0;
                  0 0 1 1 1 1 1 1 1 1 0 0;
                  0 0 0 1 1 1 1 1 1 0 0 0;
                  0 0 0 0 0 1 1 0 0 0 0 0];

romb = [0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 1 1 0 0 0 0 0;
        0 0 0 0 1 1 1 1 0 0 0 0;
        0 0 0 1 1 1 1 1 1 0 0 0;
        0 0 1 1 1 1 1 1 1 1 0 0;
        0 0 1 1 1 1 1 1 1 1 0 0;
        0 0 1 1 1 1 1 1 1 1 0 0;
        0 0 1 1 1 1 1 1 1 1 0 0;
        0 0 0 1 1 1 1 1 1 0 0 0;
        0 0 0 0 1 1 1 1 0 0 0 0;
        0 0 0 0 0 1 1 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0];

masks = cell(1, 5); 
effective_repro_lab = cell(1, 5); % Cell array to hold the 5 specific palettes
angles = [0, 45, 90, 135];

% Calculate masks and specific brightness ratios for all 5 shapes
for k = 1:4
    masks{k} = logical(imrotate(bar_12, angles(k), 'nearest', 'crop'));
    current_ratio = sum(masks{k}(:)) / 144; 
    temp_rgb = repro_palette * current_ratio;
    temp_3d = reshape(temp_rgb, [actual_num_colors, 1, 3]);
    effective_repro_lab{k} = reshape(rgb2lab(temp_3d), [actual_num_colors, 3]);
end

masks{5} = logical(circle_outlier);
current_ratio = sum(masks{5}(:)) / 144; 
temp_rgb = repro_palette * current_ratio;
temp_3d = reshape(temp_rgb, [actual_num_colors, 1, 3]);
effective_repro_lab{5} = reshape(rgb2lab(temp_3d), [actual_num_colors, 3]);

%% Visualize
figure;
subplot(1,2,1); 
imshow(img_org); % Fixed variable name
title('Original Image');

subplot(1,2,2); 
% Create a swatch using the actual number of extracted colors
% repelem creates nice clean blocks without interpolation blur
swatch_blocks = repelem(repro_palette, 20, 20); 
swatch_img = reshape(swatch_blocks, actual_num_colors * 20, 20, 3);

imshow(swatch_img); 
title(sprintf('Optimized Subset (%d Colors)', actual_num_colors));

%% create new image
clc
img_final = zeros(img_hight, img_width, 3);

for i = 1:(img_hight/12)
    for j = 1:(img_width/12)
        
        row_start = 1 + (i-1)*12;
        row_end   = row_start + 11;
        col_start = 1 + (j-1)*12;
        col_end   = col_start + 11;
        
        current_block = img_work(row_start:row_end, col_start:col_end, :);
        block_r = current_block(:,:,1);
        block_g = current_block(:,:,2);
        block_b = current_block(:,:,3);

        gray_block = rgb2gray(current_block); 
        [Gmag, Gdir] = imgradient(gray_block); 
        
        avg_mag = mean(Gmag(:)); 
        edge_threshold = 0.2;
        
        if avg_mag < edge_threshold % low magnitude: use circle
            chosen_k = 5; 

        else % high magnitude: find right angle
            valid_dirs = Gdir(Gmag > max(Gmag(:))*0.5);
            
            if isempty(valid_dirs) % Failsafe if no edges are found
                chosen_k = 5; 
            else
                rads = deg2rad(valid_dirs);
                
                % calculate average direction based of sin and cos
                avg_dir = rad2deg(atan2(mean(sin(rads)), mean(cos(rads))));
                                
                % Gradient is perpendicular compaired to the edge. Add 90 degrees to follow the edge.
                % mod(..., 180) converts the interval [-180, 180] to [0, 180)
                edge_dir = mod(avg_dir + 90, 180);
                
                angles_to_check = [0, 45, 90, 135, 180];
                [~, min_idx] = min(abs(angles_to_check - edge_dir));
                if min_idx == 5
                    min_idx = 1; % 180 same as 0 degrees
                end
                chosen_k = min_idx;
            end
        end
        
        % Color match the selected figure
        current_mask = masks{chosen_k};
        avg_r = mean(block_r(current_mask));
        avg_g = mean(block_g(current_mask));
        avg_b = mean(block_b(current_mask));
        
        % change to lab
        avg_lab = squeeze(rgb2lab(reshape([avg_r, avg_g, avg_b], 1, 1, 3)))';
        
        % find closest color in our color palette
        distances = sqrt(sum((effective_repro_lab{chosen_k} - avg_lab).^2, 2));
        [~, closest_idx] = min(distances);
        
        winning_mask = current_mask;
        winning_color = repro_palette(closest_idx, :);
        
        % Paint the winner
        for c = 1:3
            canvas = img_final(row_start:row_end, col_start:col_end, c);
            canvas(winning_mask) = winning_color(c);
            img_final(row_start:row_end, col_start:col_end, c) = canvas;
        end
    end
end
%% save
imwrite(img_final,"test.png")
%imwrite(img_work,"-original.png")
%% 6. Visualize Final Results
figure('Name', 'Mosaic Results', 'Position', [100, 100, 1200, 500]);
subplot(1,2,1);
imshow(img_work);
title('Original Image');
subplot(1,2,2); 
imshow(img_final);
title(sprintf('Dynamic Shape Mosaic (%d Colors)', actual_num_colors));
%% Evaluation
clc

ref_img = im2double(img_work); 
test_img = im2double(img_final);

% --- MÅTT 1: PSNR ---
psnr_val = psnr(test_img, ref_img);
fprintf('PSNR: %.2f dB\n', psnr_val);

% --- MÅTT 2: SSIM ---
ssim_val = ssim(test_img, ref_img);
fprintf('SSIM: %.4f\n', ssim_val);

% --- MÅTT 3: S-CIELAB ---
samplePerDeg = 81 * 118 * tan(pi/180);
scielab_val = scielab(samplePerDeg, rgb2xyz(ref_img), rgb2xyz(test_img), [65.05, 100, 108.9], 'xyz');
fprintf('S-CIELAB (Mean Delta E): %.4f\n', mean(scielab_val(:)));











