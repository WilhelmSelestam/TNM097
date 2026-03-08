%%% Project
clc
clear
% Generate color palette
% 1. 16 Standard ANSI colors
ansi = [0,0,0; 128,0,0; 0,128,0; 128,128,0; 0,0,128; 128,0,128; 0,128,128; 192,192,192;
        128,128,128; 255,0,0; 0,255,0; 255,255,0; 0,0,255; 255,0,255; 0,255,255; 255,255,255] / 255;

% 2. 216 colors (the 6x6x6 color cube)
vals = [0, 95, 135, 175, 215, 255] / 255;
[R, G, B] = ndgrid(vals, vals, vals);
cube = [R(:), G(:), B(:)];

% 3. 24 Grayscale levels (gradient from dark to light)
grays_val = linspace(8, 238, 24)' / 255;
grays = [grays_val, grays_val, grays_val];

% Combine into the final 256x3 dataset
my_256_palette = [ansi; cube; grays];
my_256_palette = my_256_palette(1:256, :); % Final array

%% Visualize the colors 
% 1. Reshape the 256x3 palette into a 16x16 grid
% Since each color is an RGB triplet, the final size is 16x16x3
palette_grid = reshape(my_256_palette, [16, 16, 3]);

% 2. Display the grid
figure;
imshow(palette_grid, 'InitialMagnification', 'fit');
axis on;
title('256 Color Dataset (16x16 Grid)');

% Optional: Label the axes to help identify specific color indices
xlabel('Column');
ylabel('Row');

%% Load Image and get its main color clusters
clc
%img_org = im2double(imread("peppers_color.tif"));
img_org = im2double(imread("lake.jpg"));
%img_org = im2double(imread("sea_sky.jpg"));

% Get dimenssions of input image
[rows, cols, channels] = size(img_org);

% get number of figures in col/row
num_fig_hight = floor(720/12);
num_fig_width = floor((720 * cols/rows)/12);

img_work = imresize(img_org, [num_fig_hight*12,num_fig_width*12], "nearest");

% Convert original image to CIELAB
img_lab = rgb2lab(img_work);

% Flatten to Nx3 list of pixels
pixel_data_lab = reshape(img_lab, [], 3);

% Shapes

% 15x15
romb_15 = [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
           0 0 0 0 0 0 1 1 1 0 0 0 0 0 0;
           0 0 0 0 0 1 1 1 1 1 0 0 0 0 0;
           0 0 0 0 1 1 1 1 1 1 1 0 0 0 0;
           0 0 0 1 1 1 1 1 1 1 1 1 0 0 0;
           0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
           0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
           0 0 0 1 1 1 1 1 1 1 1 1 0 0 0;
           0 0 0 0 1 1 1 1 1 1 1 0 0 0 0;
           0 0 0 0 0 1 1 1 1 1 0 0 0 0 0;
           0 0 0 0 0 0 1 1 1 0 0 0 0 0 0;
           0 0 0 0 0 0 0 1 0 0 0 0 0 0 0];

%% 3. Subset Selection (K-means)
num_colors = 64; 

% Create a SINGLE standard matrix to find our bright colors
representative_ratio = 0.6; % Rough average ratio of our shapes
darken_palette_rgb = my_256_palette * representative_ratio;
darken_palette_3d = reshape(darken_palette_rgb, [256, 1, 3]);
darken_palette_lab = reshape(rgb2lab(darken_palette_3d), [256, 3]);

% K-Means Clustering
[~, image_centroids_lab] = kmeans(pixel_data_lab, num_colors, 'MaxIter', 200, 'EmptyAction', 'drop');
idx = knnsearch(darken_palette_lab, image_centroids_lab);

unique_idx = unique(idx);
repro_palette = my_256_palette(unique_idx, :);
actual_num_colors = size(repro_palette, 1);

% 4. Define Shapes and their specific Palettes
% Bar with colors at rows 4 to 12
bar_15 = zeros(12, 12);
bar_15(4:11, 2:11) = 1;

% Square with a 1-pixel border of zeros
%square_outlier = zeros(15, 15); 
% square_outlier = [0 0 0 0 0 1 1 1 1 1 0 0 0 0 0;
%                   0 0 0 0 1 1 1 1 1 1 1 0 0 0 0;
%                   0 0 0 1 1 1 1 1 1 1 1 1 0 0 0;
%                   0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
%                   0 1 1 1 1 1 1 1 1 1 1 1 1 1 0;
%                   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%                   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%                   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%                   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%                   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%                   0 1 1 1 1 1 1 1 1 1 1 1 1 1 0;
%                   0 0 1 1 1 1 1 1 1 1 1 1 1 0 0;
%                   0 0 0 1 1 1 1 1 1 1 1 1 0 0 0;
%                   0 0 0 0 1 1 1 1 1 1 1 0 0 0 0;
%                   0 0 0 0 0 1 1 1 1 1 0 0 0 0 0];

square_outlier = [0 0 0 0 1 1 1 1 0 0 0 0;
                  0 0 0 1 1 1 1 1 1 0 0 0;
                  0 0 1 1 1 1 1 1 1 1 0 0;
                  0 1 1 1 1 1 1 1 1 1 1 0;
                  1 1 1 1 1 1 1 1 1 1 1 1;
                  1 1 1 1 1 1 1 1 1 1 1 1;
                  1 1 1 1 1 1 1 1 1 1 1 1;
                  0 1 1 1 1 1 1 1 1 1 1 1;
                  0 0 1 1 1 1 1 1 1 1 1 0;
                  0 0 1 1 1 1 1 1 1 1 0 0;
                  0 0 0 1 1 1 1 1 1 0 0 0;
                  0 0 0 0 1 1 1 1 0 0 0 0];

masks = cell(1, 5); 
effective_repro_lab = cell(1, 5); % Cell array to hold the 5 specific palettes
angles = [0, 45, 90, 135];

% Calculate masks and specific brightness ratios for all 5 shapes
for k = 1:4
    masks{k} = logical(imrotate(bar_15, angles(k), 'nearest', 'crop'));
    current_ratio = sum(masks{k}(:)) / 144; 
    temp_rgb = repro_palette * current_ratio;
    temp_3d = reshape(temp_rgb, [actual_num_colors, 1, 3]);
    effective_repro_lab{k} = reshape(rgb2lab(temp_3d), [actual_num_colors, 3]);
end

masks{5} = logical(square_outlier);
current_ratio = sum(masks{5}(:)) / 144; 
temp_rgb = repro_palette * current_ratio;
temp_3d = reshape(temp_rgb, [actual_num_colors, 1, 3]);
effective_repro_lab{5} = reshape(rgb2lab(temp_3d), [actual_num_colors, 3]);

% Visualize
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
img_final = zeros(num_fig_hight * 12, num_fig_width * 12, 3);

for i = 1:num_fig_hight
    for j = 1:num_fig_width
        
        row_start = 1 + (i-1)*12;
        row_end   = row_start + 11;
        col_start = 1 + (j-1)*12;
        col_end   = col_start + 11;
        
        current_block = img_work(row_start:row_end, col_start:col_end, :);
        block_r = current_block(:,:,1);
        block_g = current_block(:,:,2);
        block_b = current_block(:,:,3);
        
        best_error = inf;
        winning_mask = masks{1};
        winning_color = repro_palette(1, :);
        
        % The 5-Shape Tournament
        for k = 1:5
            current_mask = masks{k};
            avg_r = mean(block_r(current_mask));
            avg_g = mean(block_g(current_mask));
            avg_b = mean(block_b(current_mask));
            
            current_block_rgb = zeros(1,1,3);
            current_block_rgb(1,1,1) = avg_r;
            current_block_rgb(1,1,2) = avg_g;
            current_block_rgb(1,1,3) = avg_b;
            current_block_lab = rgb2lab(current_block_rgb);
            avg_lab = squeeze(current_block_lab)';
            
            % Compare against this specific shape's effective palette
            distances = sqrt(sum((effective_repro_lab{k} - avg_lab).^2, 2));
        
            [min_dist, closest_idx] = min(distances);
            
            if min_dist < best_error
                best_error = min_dist;               
                winning_mask = current_mask;         
                winning_color = repro_palette(closest_idx, :); 
            end
        end
        
        % Paint the winner
        for c = 1:3
            canvas = img_final(row_start:row_end, col_start:col_end, c);
            canvas(winning_mask) = winning_color(c);
            img_final(row_start:row_end, col_start:col_end, c) = canvas;
        end
    end
end

%% 6. Visualize Final Results
figure('Name', 'Mosaic Results', 'Position', [100, 100, 1200, 500]);
subplot(1,2,1); 
imshow(img_org); 
title('Original Image');
subplot(1,2,2); 
imshow(img_final);
title(sprintf('Dynamic Shape Mosaic (%d Colors)', actual_num_colors));