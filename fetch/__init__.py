from gym.envs.registration import register

from fetch.gym_fetch import GymFetchEnv
from fetch.stack_two import StackTwo
from fetch.no_lid import BoxNoLidEnv
from fetch.box import BoxEnv
from fetch.bin import BinEnv
from fetch.drawer import DrawerEnv
from fetch.mixed_envs.box_block import BoxBlockEnv
from fetch.mixed_envs.twin_box import TwinBoxEnv
from fetch.mixed_envs.drawer_block import DrawerBlockEnv
from fetch.mixed_envs.box_bin import BoxBinEnv
from fetch.mixed_envs.drawer_bin import DrawerBinEnv
from fetch.mixed_envs.box_bin_drawer import BoxBinDrawerEnv
from fetch.wrappers import SampleEnv

kw = dict(max_episode_steps=50, )
# original gym envs
for action in ['reach', 'push', 'pick-place', 'slide']:
    register(  # Same as FetchPickAndPlace, with a bin.
        id=f"{action.title().replace('-', '')}-v0",
        entry_point=GymFetchEnv, kwargs=dict(action=action, ), **kw)
# Fetch

# ------------------------ Finalized ------------------------
# Bin Environments Bin + object, no lid
register(id='Bin-pick-v0', entry_point=BinEnv, kwargs=dict(action="pick", obs_keys=['object0', 'bin@pos']), **kw)
register(id='Bin-place-v0', entry_point=BinEnv, kwargs=dict(action="place+air", obs_keys=['object0', 'bin@pos']), **kw)

# ------------------------ Latent Planning Envs ------------------------
# ------------ Original Box single task Debug Environments -------------
register(id='Box-fixed-v0', entry_point=BoxNoLidEnv, kwargs=dict(
    freeze_objects=['box'],
    initial_qpos={'box:joint': [1.1, 0.75, 0.6, 0, 0., 0., 0.],
                  'object0:joint': [1.25, 0.75, 0.6, 0, 0., 0., 0.]},
), **kw)
register(id='Box-fixed-place-v0', entry_point=BoxNoLidEnv, kwargs=dict(
    freeze_objects=['box'],
    initial_qpos={'box:joint': [1.1, 0.75, 0.6, 0, 0., 0., 0.],
                  'object0:joint': [1.25, 0.75, 0.6, 0, 0., 0., 0.]},
    goal_sampling={'object0': dict(target="box", range=0, offset=[0, 0, 0.04], track=True)}
), **kw)
register(id='Box-fixed-place-train-v0', entry_point=SampleEnv,
         kwargs={'fetch:Box-fixed-v0': 0.8, 'fetch:Box-fixed-place-v0': 0.2, }, **kw)

register(id='Box-aside-v0', entry_point=BoxNoLidEnv, kwargs=dict(
    freeze_objects=['box'],
    initial_qpos={'box:joint': [1.25, 0.53, 0.6, 0, 0., 0., 0.],
                  'object0:joint': [1.25, 0.73, 0.6, 0, 0., 0., 0.]}
), **kw)
register(id='Box-aside-place-v0', entry_point=BoxNoLidEnv, kwargs=dict(
    freeze_objects=['box'],
    initial_qpos={'box:joint': [1.25, 0.53, 0.6, 0, 0., 0., 0.],
                  'object0:joint': [1.25, 0.73, 0.6, 0, 0., 0., 0.]},
    goal_sampling={'object0': dict(target="box", range=0, offset=[0, 0, 0.04], track=True)}
), **kw)
register(id='Box-aside-place-train-v0', entry_point=SampleEnv,
         kwargs={'fetch:Box-aside-v0': 0.8, 'fetch:Box-aside-place-v0': 0.2, }, **kw)

# ----------------------- Debugging -----------------------
# Debug Environments todo: run these then remove
# Same as FetchPickAndPlace
register(id='Bin-no-bin-v0', entry_point=BinEnv, kwargs=dict(action='no-bin', ), **kw)
# Same as FetchPickAndPlace
register(id='Bin-pp-xml-v0', entry_point=BinEnv, kwargs=dict(action='pp-xml', ), **kw)
# Same as FetchPickAndPlace, but with a bin in model (not shown)
register(id='Bin-no-init-v0', entry_point=BinEnv, kwargs=dict(action='no-init', ), **kw)
# Bin is welded in-place
register(id='Bin-aside-hidden-v0', entry_point=BinEnv, kwargs=dict(action="bin-aside", obs_keys=['object0']), **kw)
register(id='Bin-aside-v0', entry_point=BinEnv, kwargs=dict(action="bin-aside", obs_keys=['object0', 'bin@pos']), **kw)
register(id='Bin-fixed-v0', entry_point=BinEnv, kwargs=dict(action="bin-fixed", obs_keys=['object0', 'bin@pos']), **kw)

# Drawer Debug Tasks # todo: These are currently the same as the Drawer place tasks.
register(id='Drawer-fixed-v0', entry_point=DrawerBlockEnv, kwargs=dict(action="open+place", ), **kw)
register(id='Drawer-fixed-open-v0', entry_point=DrawerBlockEnv, kwargs=dict(action="place", ), **kw)
register(id='Drawer-fixed-mixed-v0', entry_point=DrawerBlockEnv, kwargs=dict(action="mixed+place", ), **kw)


# ---------------- Latent Planning Task Set -----------------
def vec_stack_two(**kwargs):
    from fetch.wrappers import HERVecGoal
    env = StackTwo(**kwargs)
    env = HERVecGoal(env, goal_keys=['object0', 'object1'])
    return env


# [1.34193226 0.74910037 0.53472284]


register(id='StackTwo-train-v0', entry_point=vec_stack_two, kwargs=dict(action="train", ), **kw)
# we fix the first object in the center
register(id='StackTwo-fixed-v0', entry_point=vec_stack_two, kwargs=dict(action="fix-obj0-center"), **kw)
register(id='StackTwo-fixed-pp-goals-v0', entry_point=vec_stack_two, kwargs=dict(action="fix-obj0-pp-goals"), **kw)
register(id='StackTwo-v0', entry_point=vec_stack_two, kwargs=dict(action="test", ), **kw)


def vec_block_env(**kwargs):
    from fetch.wrappers import HERVecGoal
    env = BoxBlockEnv(**kwargs)
    env = HERVecGoal(env)
    return env


# The original setting (not working)
# register(id='Box-fixed-open-v0', entry_point=vec_goal_env,
#          kwargs=dict(action="open@box-fixed@obj-fixed", goal_key=("object0", "lid")), **kw)
# register(id='Box-fixed-close-v0', entry_point=vec_goal_env,
#          kwargs=dict(action="close@box-fixed@obj-fixed", goal_key=("object0", "lid")), **kw)
# register(id='Box-fixed-place-easy-v0', entry_point=vec_goal_env,
#          kwargs=dict(action="place@box-fixed@lid-fixed", goal_key=("object0", "lid")), **kw)
# with debug flag
# v0 envs: with tracking.
# v1,2,3 envs: without tracking.
register(id='Box-fixed-open-v0', entry_point=BoxBlockEnv,
         kwargs=dict(action="open", goal_key=("lid",),
                     freeze_objects=("box",),
                     obj_reset={'lid': dict(track='box', range=0, avoid=None)},
                     goal_sampling={'lid': dict(target="box", offset=[0, 0, 0.08]),
                                    # 'object0': dict(target='object0')
                                    }), **kw)
register(id='Box-fixed-close-v0', entry_point=BoxBlockEnv,
         kwargs=dict(action="close", goal_key=("lid",),
                     freeze_objects=("box",),
                     obj_reset={'lid': dict(avoid=['box', 'gripper'], d_min=0.075)},
                     goal_sampling={'lid': dict(target="box", track=True),
                                    # 'object0': dict(target='object0')
                                    }), **kw)
register(id='Box-fixed-place-easy-v0', entry_point=BoxBlockEnv,
         kwargs=dict(action="place", goal_key=("object0",),
                     freeze_objects=("box",),
                     obj_reset={'lid': dict(avoid=['box', 'gripper'], d_min=0.075),
                                'object0': dict(avoid=['lid'], d_min=0.2)},
                     goal_sampling={
                         'object0': dict(target='box', offset=[0, 0, 0.05], track=True),
                         # 'lid': dict(high=0)
                     }),
         **kw)
# disable the lid target in this version to debug
# register(id='Box-fixed-open-v0', entry_point=BoxBlockEnv,
#          kwargs=dict(action="open@box-fixed", goal_key="lid"), **kw)
# register(id='Box-fixed-close-v0', entry_point=BoxBlockEnv,
#          kwargs=dict(action="close@box-fixed", goal_key="lid"), **kw)
# register(id='Box-fixed-place-easy-v0', entry_point=BoxBlockEnv,
#          kwargs=dict(action="place@box-fixed", goal_key="object0"), **kw)
# register(id='Box-fixed-place-medium-v0', entry_point=vec_block_env,
#          kwargs=dict(action="open+place@box-fixed", goal_key=("object0", "lid")), **kw)
# register(id='Box-fixed-place-v0', entry_point=vec_block_env,
#          kwargs=dict(action="open+place+close@box-fixed", goal_key=("object0", "lid"),
#                      goal_sampling={'lid': dict(target="box", offset=[0, 0, 0.1], track=True)}), **kw)

# Box Environments: Box + Lid, there is no object
register(id='Box-open-v0', entry_point=BoxEnv, kwargs=dict(action="open", ), **kw)
register(id='Box-close-v0', entry_point=BoxEnv, kwargs=dict(action="close", ), **kw)
# Box + Object Environments, w/ additional goal for the lid
register(id='Box-place-easy-v0', entry_point=BoxBlockEnv, kwargs=dict(
    action="place", obj_keys=['box', 'lid', 'object0'],
    obj_reset={'lid': dict(avoid="box", d_min=0.1, range=0.2),
               'object0': dict(avoid=["box", "lid"], d_min=0.1, range=0.2)}
), **kw)
register(id='Box-place-medium-v0', entry_point=vec_block_env,
         kwargs=dict(action="open+place", goal_key=("object0", "lid"),
                     obj_keys=['box', 'lid', 'object0'],
                     obj_reset={'lid': dict(track="box", offset=[0, 0, 0.08], range=0),
                                'object0': dict(avoid=["box", "lid"], d_min=0.1, range=0.2)},
                     goal_sampling={'object0': dict(target='box', offset=[0, 0, 0.05], track=True),
                                    'lid': dict(high=0)}
                     ), **kw)
register(id='Box-place-v0', entry_point=BoxBlockEnv,
         kwargs=dict(action="open+place+close", goal_key="object0",
                     obj_keys=['box', 'lid', 'object0'],
                     obj_reset={'lid': dict(track="box", offset=[0, 0, 0.08], range=0),
                                'object0': dict(avoid=["box", "lid"], d_min=0.1)},
                     goal_sampling={'object0': dict(target='box', offset=[0, 0, 0.05], track=True),
                                    'lid': dict(high=0)}
                     ), **kw)

# -------------------- Twin Box Taskset --------------------
# fix the location of the bins
# register(id='TwinBox-pick-v0', entry_point=TwinBoxEnv, kwargs=dict(action="pick", ), **kw)
register(id='TwinBox-place-single-v0', entry_point=TwinBoxEnv, kwargs=dict(action="place", ), **kw)
register(id='TwinBox-red-v0', entry_point=TwinBoxEnv, kwargs=dict(action="place-red", ), **kw)
register(id='TwinBox-blue-v0', entry_point=TwinBoxEnv, kwargs=dict(action="place-blue", ), **kw)
register(id='TwinBox-mixed-v0', entry_point=TwinBoxEnv, kwargs=dict(action="place-mixed", ), **kw)
register(id='TwinBox-place-v0', entry_point=TwinBoxEnv, kwargs=dict(action="place-both", ), **kw)

# Drawer Environments
register(id='Drawer-open-v0', entry_point=DrawerEnv, kwargs=dict(action="open", ), **kw)
register(id='Drawer-close-v0', entry_point=DrawerEnv, kwargs=dict(action="close", ), **kw)
# these two are under construction (same as the debug env right now)
register(id='Drawer-place-easy-v0', entry_point=DrawerBlockEnv, kwargs=dict(action="place", ), **kw)
register(id='Drawer-place-v0', entry_point=DrawerBlockEnv, kwargs=dict(action="open+place", ), **kw)

# 3-body Environments
register(id='BoxBin-v0', entry_point=BoxBinEnv, kwargs=dict(action=None, ), **kw)
register(id='DrawerBin-v0', entry_point=DrawerBinEnv, kwargs=dict(action=None, ), **kw)
register(id='BoxBinDrawer-v0', entry_point=BoxBinDrawerEnv, kwargs=dict(action=None, ), **kw)
