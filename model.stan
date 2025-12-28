data {
  int<lower=1> N;             // 全データ数
  int<lower=1> J;             // 問題数 (Items)
  // 修正箇所: lower=0, upper=1 と明示する必要があります
  array[N] int<lower=0, upper=1> y;  // 正答(1)/誤答(0)
  array[N] int<lower=1> jj;   // 各データの問題ID
  
  vector[N] log_k;            // log(1 + 累積回数)
  vector[N] log_t;            // log(1 + 経過時間)
}

parameters {
  // グローバルパラメータ
  real<lower=0> beta;         // 全体的な忘却率
  
  // 階層モデルのハイパーパラメータ
  vector[2] mu;               // [平均難易度, 平均識別力(log)]
  vector<lower=0>[2] tau;     // スケール
  cholesky_factor_corr[2] L_Omega; // 相関行列のコレスキー分解
  
  // 問題ごとのパラメータ（標準化された空間でサンプリング）
  matrix[2, J] z; 
}

transformed parameters {
  // 実際に使用する問題ごとのパラメータ
  array[J] real d;
  array[J] real a;
  
  // 非中心化パラメータ化 (Non-centered parameterization)
  // z ~ Normal(0, 1) から相関を持つパラメータを生成
  // param_matrix = mu + (tau * L_Omega) * z
  matrix[2, J] params = rep_matrix(mu, J) + diag_pre_multiply(tau, L_Omega) * z;
  
  for (j in 1:J) {
    d[j] = params[1, j];
    a[j] = exp(params[2, j]); // logスケールから戻す
  }
}

model {
  // 事前分布 (Weakly Informative Priors)
  beta ~ normal(0.5, 0.5);
  mu ~ normal(0, 1);
  tau ~ cauchy(0, 2);
  L_Omega ~ lkj_corr_cholesky(2.0); // 相関行列への事前分布
  to_vector(z) ~ std_normal();      // zは標準正規分布
  
  // 尤度計算
  vector[N] S;
  vector[N] logit_p;
  
  for (n in 1:N) {
    // 記憶強度 = (学習効果) - (忘却効果)
    S[n] = log_k[n] - beta * log_t[n];
    
    // ロジット = 識別力 * 記憶強度 - 難易度
    logit_p[n] = a[jj[n]] * S[n] - d[jj[n]];
  }
  
  y ~ bernoulli_logit(logit_p);
}

generated quantities {
  // 推定結果の相関行列を復元
  matrix[2, 2] Omega;
  Omega = multiply_lower_tri_self_transpose(L_Omega);
}