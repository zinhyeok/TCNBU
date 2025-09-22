# data_generator.py

import numpy as np
from scipy.stats import cauchy

def generate_single_data(n=100, d=50, scenario='mean', seed=0):
    np.random.seed(42+seed)
    delta_sigma_tables = {
        'mean': {
            10:   (0.8, 1.0),
            50:   (1.0, 1.0),
            100:  (1.2, 1.0),
            150:  (1.6, 1.0),
            175:  (2.0, 1.0),
            500:  (2.5, 1.0),
            2000: (3.4, 1.0),
        },
        'both': {
            10:   (0.6, 1.3),
            50:   (1.0, 1.3),
            100:  (1.2, 1.1),
            150:  (1.2, 1.1),
            175:  (1.05, 1.1),
            500:  (1.0, 1.1),
            2000: (1.0, 1.05),
        },
        'scale': {
            10:   (1.0, 1.7),
            20:   (1.0, 1.5),
            50:   (1.0, 1.3),
            75:   (1.0, 1.28),
            100:  (1.0, 1.28),
            500:  (1.0, 1.1),
            2000: (1.0, 1.05),
        }
    }
    delta, scale = delta_sigma_tables[scenario][d]

    tau = (n // 4) * 2
    Sigma = np.eye(d)
    if scenario == 'mean':
        mu = np.zeros(d)
        mu[:d//5] = delta / np.sqrt(d//5)
        group1 = np.random.multivariate_normal(np.zeros(d), Sigma, size=tau)
        group2 = np.random.multivariate_normal(mu, Sigma, size=n - tau)
    elif scenario == 'scale':
        Sigma2 = Sigma.copy()
        Sigma2[:d//5, :d//5] *= scale**2
        group1 = np.random.multivariate_normal(np.zeros(d), Sigma, size=tau)
        group2 = np.random.multivariate_normal(np.zeros(d), Sigma2, size=n - tau)
    elif scenario == 'both':
        mu = np.zeros(d)
        mu[:d//5] = delta / np.sqrt(d//5)
        Sigma2 = Sigma.copy()
        Sigma2[:d//5, :d//5] *= scale**2
        group1 = np.random.multivariate_normal(np.zeros(d), Sigma, size=tau)
        group2 = np.random.multivariate_normal(mu, Sigma2, size=n - tau)
    return np.vstack((group1, group2)), tau


def generate_multi_data(scenario='model1', n=200, frequency=50, d=100, seed=0):
    """
    다양한 시나리오에 따라 다중 변화점 데이터를 생성합니다.

    Args:
        scenario (str): 데이터 생성 시나리오 ('model5', 'model52', 'model6', 'model7', 'model8').
        n (int): 총 샘플 수.
        frequency (int): 'model5', 'model6', 'model7'에서 변화점 발생 빈도.
        d (int): 데이터의 차원 (dimensionality).
        seed (int): 난수 생성을 위한 시드.
    Returns:
        tuple: 생성된 데이터 (np.ndarray)와 실제 변화점 리스트 (list).
    """


    np.random.seed(42 + seed)
    Σ = np.identity(d)


    if scenario == 'modelsingle1-1':
        T = n - 1
        cps = [100]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d, 4 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'modelsingle1-2':
        T = n - 1
        cps = [100]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d, 3 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)
            
        return np.vstack(blocks), cps

    elif scenario == 'modelsingle1-3':
        T = n - 1
        cps = [100]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d, 2 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'modelsingle2-1':
        T = n - 1
        cps = [50, 100]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,4 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'modelsingle2-2':
        T = n - 1
        cps = [50, 100]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,3 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'modelsingle2-3':
        T = n - 1
        cps = [50, 100]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,2 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps

    elif scenario == 'modelsingle3-1':
        T = n - 1
        cps = [50, 100, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,4 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'modelsingle3-2':
        T = n - 1
        cps = [50, 100, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,3 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'modelsingle3-3':
        T = n - 1
        cps = [50, 100, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,2 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps

    elif scenario == 'model1':
        T = n - 1
        cps = [50, 100, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,5 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        
        return np.vstack(blocks), cps

    elif scenario == 'model2':
        T = n - 1
        cps = [50, 75, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,5 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps

    elif scenario == 'model3':
        T = n - 1
        cps = [50, 75, 100, 125, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,5 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        
        return np.vstack(blocks), cps

    elif scenario == 'model4':
        T = n - 1
        cps = [25, 75, 100, 150, 175]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,5 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
# 코시분포간 변화     
    elif scenario == 'model5':
        T = n - 1
        cps = [50, 100, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            if i % 2 == 0:
                # Cauchy_d(0, I)
                samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
            else:
                # Cauchy_d((7/(4log(d)))*1, I) 생성
                loc_val = 7 / (4 * np.log(d))
                samples = cauchy.rvs(loc=loc_val, scale=1, size=(size, d))
            blocks.append(samples)

        return np.vstack(blocks), cps

    elif scenario == 'model6':
        T = n - 1
        cps = [50, 75, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            if i % 2 == 0:
                # Cauchy_d(0, I)
                samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
            else:
                # Cauchy_d((7/(4log(d)))*1, I) 
                loc_val = 7 / (4 * np.log(d))
                samples = cauchy.rvs(loc=loc_val, scale=1, size=(size, d))
            blocks.append(samples)

        return np.vstack(blocks), cps

    elif scenario == 'model7':
        T = n - 1
        cps = [50, 75, 100, 125, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            if i % 2 == 0:
                # Cauchy_d(0, I)
                samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
            else:
                # Cauchy_d((7/(4log(d)))*1, I)
                loc_val = 7 / (4 * np.log(d))
                samples = cauchy.rvs(loc=loc_val, scale=1, size=(size, d))
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'model8':
        T = n - 1
        cps = [25, 75, 100, 150, 175]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            if i % 2 == 0:
                # Cauchy_d(0, I)
                samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
            else:
                # Cauchy_d((7/(4log(d)))*1, I)
                loc_val = 7 / (4 * np.log(d))
                samples = cauchy.rvs(loc=loc_val, scale=1, size=(size, d))
            blocks.append(samples)

        return np.vstack(blocks), cps


# cauchy 분포 & 정규분포를 사용한 모델
    elif scenario == 'model9':
        T = n - 1
        cps = [50, 100, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []
        num_ones = d // 5
        theta = np.concatenate([np.ones(num_ones), np.zeros(d - num_ones)])
        Sigma = np.fromfunction(lambda j, k: 0.8 ** np.abs(j - k), (d, d))

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            if i % 2 == 0:
                samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
            else:
            # 구간 2: [51, 65] -> N_d( (7/log(d))*theta, Σ )
                mean = (7 / np.log(d)) * theta
                samples = np.random.multivariate_normal(mean, Sigma, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps

    elif scenario == 'model10':
        T = n - 1
        cps = [50, 75, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []
        num_ones = d // 5
        theta = np.concatenate([np.ones(num_ones), np.zeros(d - num_ones)])
        Sigma = np.fromfunction(lambda j, k: 0.8 ** np.abs(j - k), (d, d))

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            if i % 2 == 0:
                samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
            else:
            # 구간 2: [51, 65] -> N_d( (7/log(d))*theta, Σ )
                mean = (7 / np.log(d)) * theta
                samples = np.random.multivariate_normal(mean, Sigma, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps

    elif scenario == 'model11':
        T = n - 1
        cps = [50, 75, 100, 125, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []
        num_ones = d // 5
        theta = np.concatenate([np.ones(num_ones), np.zeros(d - num_ones)])
        Sigma = np.fromfunction(lambda j, k: 0.8 ** np.abs(j - k), (d, d))

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            if i % 2 == 0:
                samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
            else:
            # 구간 2: [51, 65] -> N_d( (7/log(d))*theta, Σ )
                mean = (7 / np.log(d)) * theta
                samples = np.random.multivariate_normal(mean, Sigma, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps

    elif scenario == 'model12':
        T = n - 1
        cps = [25, 75, 100, 150, 175]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []
        num_ones = d // 5
        theta = np.concatenate([np.ones(num_ones), np.zeros(d - num_ones)])
        Sigma = np.fromfunction(lambda j, k: 0.8 ** np.abs(j - k), (d, d))

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            if i % 2 == 0:
                samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
            else:
            # 구간 2: [51, 65] -> N_d( (7/log(d))*theta, Σ )
                mean = (7 / np.log(d)) * theta
                samples = np.random.multivariate_normal(mean, Sigma, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'model13':
        T = n - 1
        cps = [50, 100, 110, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            mean = np.zeros(d) if i % 2 == 0 else np.full(d ,5 / (4 * np.log(d)))
            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'model14':
        T = n - 1
        cps = [50, 100, 110, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []
        num_ones = d // 5
        theta = np.concatenate([np.ones(num_ones), np.zeros(d - num_ones)])
        Sigma = np.fromfunction(lambda j, k: 0.8 ** np.abs(j - k), (d, d))

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1
            if i % 2 == 0:
                samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
            else:
            # 구간 2: [51, 65] -> N_d( (7/log(d))*theta, Σ )
                mean = (7 / np.log(d)) * theta
                samples = np.random.multivariate_normal(mean, Sigma, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'model15':
        T = n - 1
        cps = [50, 100, 110, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        # 기본 평균 변화량과 큰 평균 변화량을 각각 정의합니다.
        mean_shift_small = 5 / (4 * np.log(d))
        # 기본 변화량보다 더 큰 값으로 설정하여 변화의 크기를 조절합니다. (예: 3배)
        mean_shift_large = mean_shift_small * 3

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1

            # 구간이 짝수 번째일 경우 평균을 0으로 설정
            # 0, 100, 150
            if i % 2 == 0:
                mean = np.zeros(d)
            # 구간이 홀수 번째일 경우
            else:
                # cp=110 직후 구간(start=110)에만 큰 변화를 적용
                if start == 109:
                    mean = np.full(d, mean_shift_large)
                # 다른 홀수 번째 구간들은 기존의 작은 변화를 적용
                else:
                    mean = np.full(d, mean_shift_small)

            cov = Σ
            samples = np.random.multivariate_normal(mean, cov, size=size)
            blocks.append(samples)

        return np.vstack(blocks), cps
    
    elif scenario == 'model16':
        """
        cp=110에서 분포 변화를 더 크게 조정한 데이터 생성 함수
        """
        T = n - 1
        cps = [50, 100, 110, 150]
        cps = [cp - 1 for cp in cps]
        boundaries = [-1] + cps + [T]
        blocks = []

        num_ones = d // 5
        theta = np.concatenate([np.ones(num_ones), np.zeros(d - num_ones)])
        Sigma = np.fromfunction(lambda j, k: 0.8 ** np.abs(j - k), (d, d))

        # 기본 변화량과 큰 변화량을 결정하는 계수를 각각 정의합니다.
        shift_factor_small = 7 / np.log(d)
        # 기본 계수보다 더 큰 값으로 설정하여 변화의 크기를 조절합니다. (예: 3배)
        shift_factor_large = shift_factor_small * 3

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i]+1, boundaries[i+1]
            size = end - start + 1

            # 구간이 짝수 번째일 경우, 코시 분포에서 샘플링
            if i % 2 == 0:
                samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
            # 구간이 홀수 번째일 경우, 다변량 정규 분포에서 샘플링
            else:
                # cp=110 직후 구간(start=110)인지 확인
                if start == 109:
                    # 더 큰 변화를 주기 위해 large factor를 사용
                    mean = shift_factor_large * theta
                else:
                    # 다른 홀수 번째 구간들은 기존의 small factor를 사용
                    mean = shift_factor_small * theta
                
                samples = np.random.multivariate_normal(mean, Sigma, size=size)
            
            blocks.append(samples)

        return np.vstack(blocks), cps

    # # 비균등 변화
    # elif scenario == 'model4':
    #     total_n = n
    #     cps = [50, 100, 150]
    #     boundaries = [0] + cps + [total_n]
    #     blocks = []
    #     samples_1 = np.random.multivariate_normal(theta*0, Σ, size=boundaries[1])
    #     samples_2 = np.random.multivariate_normal(theta*0.37, Σ, size=boundaries[2] - boundaries[1])
    #     samples_3 = np.random.multivariate_normal(theta*1.11, Σ, size=boundaries[3] - boundaries[2])
    #     samples_4 = np.random.multivariate_normal(theta*4.81, Σ, size=boundaries[4] - boundaries[3])
    #     blocks.append(samples_1)
    #     blocks.append(samples_2)
    #     blocks.append(samples_3)
    #     blocks.append(samples_4)

    #     return np.vstack(blocks), cps

    # elif scenario == 'model6':
    #     total_n = n
    #     n_blocks = total_n // frequency
    #     cps = [frequency * i for i in range(1, n_blocks)]
    #     boundaries = [0] + cps + [total_n]
    #     blocks = []

    #     for i in range(len(boundaries) - 1):
    #         start, end = boundaries[i], boundaries[i+1]
    #         size = end - start

    #         if i % 2 == 0:
    #             mu = np.zeros(d)
    #             Sigma = np.eye(d)
    #         else:
    #             mu = delta * theta
    #             Sigma = np.fromfunction(lambda i, j: 0.3 ** np.abs(i - j), (d, d))

    #         Z = np.random.multivariate_normal(np.zeros(d), Sigma, size=size)
    #         W = np.random.chisquare(1, size=size).reshape(-1, 1)
    #         samples = mu + Z / np.sqrt(W)
    #         blocks.append(samples)

    #     return np.vstack(blocks), cps

    # elif scenario == 'model7':
    #     total_n = n
    #     n_blocks = total_n // frequency
    #     cps = [frequency * i for i in range(1, n_blocks)]
    #     boundaries = [0] + cps + [total_n]
    #     blocks = []

    #     for i in range(len(boundaries) - 1):
    #         start, end = boundaries[i], boundaries[i+1]
    #         size = end - start

    #         if i % 2 == 0:
    #             cov = Σ
    #         else:
    #             cov = sigma * Σ

    #         samples = np.random.multivariate_normal(np.zeros(d), cov, size=size)
    #         blocks.append(samples)

    #     return np.vstack(blocks), cps
    
# def generate_model8_data(d, n=260, seed=0):
#     """
#     첨부된 이미지의 Model 8 명세에 따라 데이터를 생성합니다.

#     Args:
#         d (int): 데이터의 차원 (dimension)
#         n (int): 생성할 데이터의 총 개수 (기본값: 260)
#         seed (int): 재현성을 위한 랜덤 시드

#     Returns:
#         tuple: (생성된 데이터 배열, 변화점 리스트)
#     """
#     # 0. 재현성을 위한 시드 설정
#     np.random.seed(42 + seed)

#     # 1. 모델의 핵심 파라미터 정의
    
#     # theta: 첫 d/5개는 1, 나머지는 0인 희소 벡터(sparse vector)
#     if d < 5:
#         raise ValueError("차원(d)은 5 이상이어야 합니다.")
#     num_ones = d // 5
#     theta = np.concatenate([np.ones(num_ones), np.zeros(d - num_ones)])

#     # Sigma(Σ): 공분산 행렬 (Σ_jk = 0.8^|j-k|)
#     Sigma = np.fromfunction(lambda j, k: 0.8 ** np.abs(j - k), (d, d))
    
#     # I: 단위 행렬(Identity Matrix)
#     I = np.identity(d)
    
#     # 2. 변화점(changepoints) 및 구간 정의
#     cps = [50, 65, 110, 160, 185]
#     total_n = n
#     boundaries = [0] + cps + [total_n]
    
#     blocks = []

#     # 3. 구간별로 순회하며 각기 다른 분포에서 데이터 생성
#     for i in range(len(boundaries) - 1):
#         start, end = boundaries[i], boundaries[i+1]
#         size = end - start
#         if size <= 0: continue

#         # 각 구간(i=0~5)에 해당하는 분포에서 샘플링
#         if i == 0:
#             # 구간 1: [1, 50] -> Cauchy_d(0, I)
#             # d개의 독립적인 표준 코시 분포에서 샘플링
#             samples = cauchy.rvs(loc=0, scale=1, size=(size, d))
        
#         elif i == 1:
#             # 구간 2: [51, 65] -> N_d( (7/log(d))*theta, Σ )
#             mean = (7 / np.log(d)) * theta
#             samples = np.random.multivariate_normal(mean, Sigma, size=size)
            
#         elif i == 2:
#             # 구간 3: [66, 110] -> Cauchy_d(0, 2I)
#             # scale 파라미터가 2인 d개의 독립적인 코시 분포
#             samples = cauchy.rvs(loc=0, scale=2, size=(size, d))
            
#         elif i == 3:
#             # 구간 4: [111, 160] -> N_d( (-5/(2log(d)))*theta, I )
#             mean = (-5 / (2 * np.log(d))) * theta
#             samples = np.random.multivariate_normal(mean, I, size=size)
            
#         elif i == 4:
#             # 구간 5: [161, 185] -> N_d( (2/log(d))*theta, Σ )
#             mean = (2 / np.log(d)) * theta
#             samples = np.random.multivariate_normal(mean, Sigma, size=size)
            
#         elif i == 5:
#             # 구간 6: [186, 260] -> Exp_d(1) - 1
#             # d개의 독립적인 지수 분포(rate=1 or scale=1)에서 샘플링 후 1을 뺌
#             samples = np.random.exponential(scale=1.0, size=(size, d)) - 1
            
#         blocks.append(samples)

#     # 4. 결과 반환
#     # 모든 블록을 하나로 합치고, 0-based 인덱스로 변화점 조정
#     final_data = np.vstack(blocks)
#     final_cps = [cp - 1 for cp in cps]
    
#     return final_data, final_cps