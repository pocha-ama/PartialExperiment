import numpy as np
import hashlib
from otree.api import Bot, Submission
from . import *

class PlayerBot(Bot):
    def play_round(self):
        # ==========================================================
        # 1. 初期ページ群（ラウンド1のみ表示されるページ）
        # ==========================================================
        if self.round_number == 1:
            yield Submission(Stand_by, dict(typing_test='テスト入力'), check_html=False)
            yield Submission(Demographic, dict(
                group_id_number=1,
                individual_id_number=self.player.id_in_group,
                gender='男性',
                age='20'
            ), check_html=False)
            yield Submission(Instruction, check_html=False)

        # ==========================================================
        # 2. 現在のタスク状態の取得
        # ==========================================================
        all_tasks = self.player.participant.vars.get('all_tasks', [])
        idx = self.player.participant.vars.get('current_task_index', 0)

        # 全問題が終了している場合
        if idx >= len(all_tasks):
            # 実際の人間は「Results」や「Finish」でストップしますが、
            # Botは NUM_ROUNDS (1000) まで裏で空回りし続ける必要があるため、
            # 以降のラウンドでは一切のページ送信（yield）を行わずに return します。
            # これにより、一瞬でラウンド1000まで到達して終了します。
            return

        current_task = all_tasks[idx]
        task_id = current_task['task_id']

        # ==========================================================
        # 3. 絶対に一致させるための「共通の文字列」を取得
        # ==========================================================
        p1 = self.player.group.get_players()[0]
        p1_task = p1.participant.vars['all_tasks'][idx]
        universal_opt1 = p1_task['option1']
        universal_opt2 = p1_task['option2']

        # ==========================================================
        # 4. 「目標終了ステップ数」と「現在のループ回数」の計算
        # ==========================================================
        seed_str = f"bot_{p1.participant.code}_{idx}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        
        target_step = np.random.poisson(1.37) + 1

        history_len = len(self.player.participant.vars.get(f'choice_task{idx}', []))
        current_loop = max(1, history_len)

        # ==========================================================
        # 5. 意思決定（わざと割るか、一致させるか）
        # ==========================================================
        if task_id == 'practice':
            my_choice = universal_opt1
            target_step = 1
        elif current_loop >= target_step:
            my_choice = universal_opt1
        else:
            my_choice = universal_opt1 if self.player.id_in_group == 1 else universal_opt2

        # ==========================================================
        # 6. 各ページの自動通過
        # ==========================================================
        if current_loop == 1:
            yield Submission(Question, check_html=False)
            yield Submission(First_Make_Decision, dict(first_decision_making=my_choice, first_confidence=5), check_html=False)

        yield Submission(Chat, check_html=False)

        if task_id == 'practice':
            yield Submission(After_Practice, check_html=False)
            return

        yield Submission(Nth_Make_Decision, dict(nth_decision_making=my_choice, nth_confidence=5), check_html=False)

        if current_loop >= target_step:
            yield Submission(Unanimity, check_html=False)