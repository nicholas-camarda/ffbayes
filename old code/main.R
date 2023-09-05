library(tidyverse)
library(readxl)
library(purrr)
library(conflicted)
library(devtools)
# library(brms)
library(gtools)
library(fitdistrplus)
library(gghighlight)
library(GetoptLong)
library(rethinking)
rstan_options(auto_write = TRUE)

conflict_prefer("filter", "dplyr")
# conflict_prefer("ar", "brms")
# conflict_prefer("ddirichlet", "brms")
conflict_prefer("select", "dplyr")

# tbl <- read_csv("~/Desktop/coding/ffbayes/datasets/snake-draft_ppr-0.5_vor_top-120_2020.csv")

db <- read_csv("~/Desktop/coding/ffbayes/datasets/2017season.csv") %>% 
  bind_rows(read_csv("~/Desktop/coding/ffbayes/datasets/2018season.csv")) %>% 
  bind_rows(read_csv("~/Desktop/coding/ffbayes/datasets/2019season.csv")) %>%
  mutate(FantPt = ifelse(is.na(FantPt), 0, FantPt),
         game = str_c("g",`G#`)) %>%
  filter(!is.na(Position))

home_team_tbl <- db %>% 
  distinct(Tm) %>%
  mutate(home_team_id = 1:n()); db <- db %>% left_join(home_team_tbl)
opp_team_tbl <- db %>% 
  distinct(Opp) %>%
  mutate(opp_team_id = 1:n()); db <- db %>% left_join(opp_team_tbl)
pos_tbl <- db %>% 
  distinct(Position) %>%
  mutate(pos_id = 1:n()); db <- db %>% left_join(pos_tbl)
playr_tbl <- db %>% 
  distinct(Name) %>%
  mutate(player_id = 1:n()) %>%
  mutate(player_id = str_c("X", as.character(player_id))); db <- db %>% left_join(playr_tbl)

db_work <- db %>% 
  select(player_id, Name, Season, pos_id, Position, game, home_team_id, opp_team_id, Away, pos_id, FantPt) %>%
  mutate(game = factor(game, str_c("g",1:16))) 


get_quantiles <- function(database = db_work, player_name = c("Todd Gurley")){
  # player_name = c("Jared Cook", "Anthony Miller)
  # player_name <- agrep(pattern = p_name, x = unique(database$Name), value=T)
  # if (length(player_name) > 1) stop("Not specific enough - please type full name.")
  dat <- database %>% filter(Name %in% player_name)
  pos_p <- dat$Position %>% unique(); pos_p
  # dens_p <- density(dat$FantPt)
  # ref <- filter(database, Name != player_name & Position == pos_p)
  # dens_r <- density(ref$FantPt)
  
  db_plot <- database %>% 
    mutate(Name = as.character(Name)) %>%
    
    mutate(Name = ifelse(Name %in% player_name, Name, Position)) %>%
    filter(Position %in% pos_p)
  
  mu <- db_plot %>% 
    group_by(Name) %>% 
    summarize(grp.mean = mean(FantPt))
  
  playeri <- ggplot(db_plot, aes(x = FantPt, color = Name)) +
    geom_density() + 
    gghighlight(Name %in% player_name, keep_scales = TRUE) +
    geom_vline(data=mu, aes(xintercept=grp.mean, color=Name),
               linetype="dashed")  + theme_minimal()
    playeri
}



get_quantiles(player_name = c("Jared Cook", "Jalen Reagor"))


explore_tbl <- db_work %>% filter(Season == "2017")
train_tbl <- db_work %>% filter(Season == "2018")
test_tbl <- db_work %>% filter(Season == "2019")

summary_tbl <- db_work %>% 
  group_by(player_id, Season) %>% 
  summarize(szn_avg = mean(FantPt, na.rm = T))


# EDA
# https://www.monicaalexander.com/posts/2020-28-02-bayes_viz/
filter(db_work, player_id %in% c("X1","X2")) %>% 
  ggplot(aes(game, FantPt, color = player_id)) + 
  geom_point() + 
  geom_smooth(method = "lm") + 
  scale_color_brewer(palette = "Set1") + 
  theme_bw(base_size = 14) +
  ggtitle("FantPt x game")


set.seed(42)
f1_tbl <- explore_tbl %>% 
  select(player_id, game, FantPt) %>%
  spread(game, FantPt) %>%
  mutate(player_id = mixedsort(player_id));

# f2_tbl <- test_tbl %>% 
#   slice(-1509) %>%
#   select(player_id, game, FantPt) %>%
#   spread(game, FantPt) %>%
#   mutate(player_id = mixedsort(player_id))

fp_playeri <- filter(f1_tbl, player_id == "X1") %>% select(-1) %>% as.numeric()
fp_playeri <- fp_playeri[!is.na(fp_playeri)]

# fp_playeri2 <- filter(f2_tbl, player_id == "X1") %>% select(-1) %>% as.numeric()
# fp_playeri2 <- fp_playeri2[!is.na(fp_playeri2)]
# f <- density(fp_playeri,na.rm = T)

