library(readxl)
library(ggplot2)
library(dplyr)
library(stringr)
library(grid)

# -----------------------------------------------------------------------------
# 配置与数据加载
# -----------------------------------------------------------------------------
FILE_PATH <- "multi_journal_analysis_report.xlsx"
OUTPUT_PDF <- "Medical_Journal_Analysis_Report_R.pdf"

# 设置主题
custom_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18, color = "#2C3E50", hjust = 0.5, margin = margin(b = 20)),
    axis.title = element_text(face = "bold", size = 12, color = "#566573"),
    axis.text = element_text(size = 10, color = "#2C3E50"),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "#EBEDEF"),
    legend.position = "none"
  )

# 颜色板 (Viridis or customized)
my_colors <- c("#2E86C1", "#1ABC9C", "#F1C40F", "#E67E22", "#E74C3C", "#8E44AD", "#34495E")

# 加载数据
message("📂 Loading data...")
df <- read_excel(FILE_PATH)

# 处理列名不一致问题
if (!"Journal" %in% colnames(df) && "Journal/Book" %in% colnames(df)) {
  df <- df %>% rename(Journal = `Journal/Book`)
}

# -----------------------------------------------------------------------------
# 绘图函数
# -----------------------------------------------------------------------------

# 1. 封面页
create_cover <- function() {
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(3, 1, heights = c(1, 1, 1))))
  
  grid.text("Medical Journal Analysis Report", y = 0.7, gp = gpar(fontsize = 30, fontface = "bold", col = "#2C3E50"))
  
  info_text <- paste0(
    "Generated on: ", Sys.Date(), "\n\n",
    "Total Papers Analyzed: ", nrow(df), "\n",
    "Data Source: ", FILE_PATH
  )
  grid.text(info_text, y = 0.4, gp = gpar(fontsize = 14, col = "#566573", lineheight = 1.5))
  
  grid.lines(x = unit(c(0.2, 0.8), "npc"), y = unit(0.2, "npc"), gp = gpar(col = "#2E86C1", lwd = 3))
}

# 2. 通用条形图
plot_bar <- function(data, x_col, y_col, title, x_lab, y_lab, fill_color = "#2E86C1", top_n = NULL) {
  if (!is.null(top_n)) {
    data <- head(data, top_n)
  }
  
  # 确保顺序正确 (ggplot2 默认按字母序，我们需要按数量倒序)
  data[[x_col]] <- factor(data[[x_col]], levels = rev(data[[x_col]]))
  
  p <- ggplot(data, aes(x = !!sym(x_col), y = !!sym(y_col))) +
    geom_col(fill = fill_color, width = 0.7) +
    geom_text(aes(label = !!sym(y_col)), hjust = -0.1, size = 3.5, color = "#566573") +
    coord_flip() +
    labs(title = title, x = x_lab, y = y_lab) +
    custom_theme +
    theme(axis.text.y = element_text(margin = margin(r = 10))) # 增加标签间距
    
  return(p)
}

# 3. 饼图
plot_pie <- function(data, group_col, count_col, title) {
  # 计算百分比
  data <- data %>%
    mutate(prop = !!sym(count_col) / sum(!!sym(count_col)) * 100) %>%
    mutate(ypos = cumsum(prop) - 0.5 * prop)
    
  p <- ggplot(data, aes(x = "", y = prop, fill = !!sym(group_col))) +
    geom_bar(stat = "identity", width = 1, color = "white") +
    coord_polar("y", start = 0) +
    theme_void() + 
    theme(
        plot.title = element_text(face = "bold", size = 18, color = "#2C3E50", hjust = 0.5, margin = margin(b = 20)),
        legend.position = "right",
        legend.title = element_blank(),
        legend.text = element_text(size = 10)
    ) +
    geom_text(aes(y = ypos, label = paste0(round(prop, 1), "%")), color = "white", size = 4) +
    labs(title = title) +
    scale_fill_brewer(palette = "Set2")
    
  return(p)
}

# 4. 热力图
plot_heatmap <- function(data) {
  heatmap_data <- data %>%
    count(Journal, `Research Design`) %>%
    group_by(Journal) %>%
    mutate(prop = n / sum(n)) %>%
    ungroup()
    
  p <- ggplot(heatmap_data, aes(x = `Research Design`, y = Journal, fill = prop)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "#EBF5FB", high = "#2E86C1") +
    geom_text(aes(label = sprintf("%.1f", prop)), color = ifelse(heatmap_data$prop > 0.5, "white", "black"), size = 3) +
    labs(title = "Research Design Distribution by Journal", x = "Research Design", y = "Journal", fill = "Proportion") +
    custom_theme +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid.major.x = element_blank(), # 热力图不需要网格
        legend.position = "right"
    )
    
  return(p)
}

# -----------------------------------------------------------------------------
# 生成流程
# -----------------------------------------------------------------------------
message("🚀 Generating plots...")

pdf(OUTPUT_PDF, width = 11.7, height = 8.3) # A4 Landscape

# P0: 封面
create_cover()

# P1: Top Journals
d1 <- df %>% count(Journal, sort = TRUE)
print(plot_bar(d1, "Journal", "n", "Top Journals by Publication Volume", "Journal", "Count", top_n = 10))

# P2: Research Design
d2 <- df %>% count(`Research Design`, sort = TRUE)
print(plot_bar(d2, "Research Design", "n", "Distribution of Research Designs", "Research Design", "Count", fill_color = "#1ABC9C", top_n = 15))

# P3: Study Timing
d3 <- df %>% count(`Study Timing`, sort = TRUE)
# 只保留前7个，其他的合并为Other
if(nrow(d3) > 7) {
    top7 <- head(d3, 7)
    other <- data.frame(`Study Timing` = "Other", n = sum(tail(d3, -7)$n))
    colnames(other) <- colnames(d3) # 修复列名不匹配
    d3 <- rbind(top7, other)
}
print(plot_pie(d3, "Study Timing", "n", "Study Timing Distribution"))

# P4: Disease System
d4 <- df %>% count(`Focused Disease System`, sort = TRUE)
print(plot_bar(d4, "Focused Disease System", "n", "Top Focused Disease Systems", "Disease System", "Count", fill_color = "#E67E22", top_n = 15))

# P5: Specific Disease
d5 <- df %>% 
    filter(`Focused Disease` != "Not Applicable") %>%
    count(`Focused Disease`, sort = TRUE)
print(plot_bar(d5, "Focused Disease", "n", "Top 20 Specific Diseases/Conditions", "Disease", "Count", fill_color = "#E74C3C", top_n = 20))

# P6: Heatmap
print(plot_heatmap(df))

# P7: Country
if("Research Team Country" %in% colnames(df)) {
    d7 <- df %>% count(`Research Team Country`, sort = TRUE)
    print(plot_bar(d7, "Research Team Country", "n", "Top Research Team Countries", "Country", "Count", fill_color = "#8E44AD", top_n = 15))
}

dev.off()

message(paste("🎉 PDF Report saved to:", OUTPUT_PDF))
