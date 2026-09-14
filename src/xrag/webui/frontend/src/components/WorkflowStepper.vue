<script setup>
defineProps({ step: Number, steps: Array });
defineEmits(['goto']);
</script>

<template>
  <div class="stepper">
    <div v-for="(label, index) in steps" :key="label" class="step"
         :class="{ 'is-complete': index + 1 < step, 'is-current': index + 1 === step }"
         @click="$emit('goto', index + 1)">
      <div class="circle">{{ index + 1 }}</div>
      <div class="label">{{ label }}</div>
    </div>
  </div>
</template>
